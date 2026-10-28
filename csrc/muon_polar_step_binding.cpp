#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "adv_api/kernel_tiling.h"
#include "adv_api/matmul/bmm_tiling.h"
#include "adv_api/transpose/confusion_transpose_tiling.h"

namespace {

/* 必须和 Cube1 low-level mixed AscendC 文件中的结构保持一致。 */
struct MuonGramLowLevelFusedTilingData {
    uint32_t batch;
    uint32_t rows;
    uint32_t common;
    uint32_t dim;
    uint32_t usedCoreNum;
    uint32_t groupsPerBatch;
    uint32_t groupTiles;
    ConfusionTransposeTiling transposeTiling128;
    ConfusionTransposeTiling transposeTiling256;
};

/* 必须和 mixed Cube3 AscendC 文件中的 MuonCube3TilingData 保持一致。 */
struct MuonCube3TilingData {
    uint32_t batch;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t mDim;
    uint32_t nDim;
    uint32_t usedCoreNum;
    uint16_t alphaBits;
    uint16_t reserved;
    TCubeTiling cubeTiling;
};

/* 必须和 Cube2+P mixed AscendC 文件中的 tiling 结构保持一致。 */
struct MuonCube2PTilingData {
    uint32_t batch;
    uint32_t rows;
    uint32_t mDim;
    uint32_t nDim;
    uint32_t usedCoreNum;
    uint32_t bBits;
    uint32_t cBits;
    uint32_t reserved;
    TCubeTiling cubeTiling;
    ConfusionTransposeTiling transposeTiling;
};

extern "C" void launch_muon_gram_lowlevel_fused(void *x, void *z,
    void *tiling, uint32_t blockDim, aclrtStream stream);
extern "C" void launch_muon_cube3_fused_lowlevel(void *polynomial, void *x,
    void *y, void *tiling, uint32_t blockDim, aclrtStream stream);
extern "C" void launch_muon_cube2_p_fused(void *a, void *p, void *tiling,
    uint32_t blockDim, aclrtStream stream);

void CheckAcl(aclError ret, const char *expr)
{
    TORCH_CHECK(ret == ACL_SUCCESS, expr, " failed, ret=", ret, ", detail=", aclGetRecentErrMsg());
}

uint64_t EnvU64(const char *name, uint64_t defaultValue)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char *end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    return end != value ? parsed : defaultValue;
}

int64_t AlignUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

void DebugSync(aclrtStream stream, const char *stage)
{
    if (std::getenv("MUON_POLAR_DEBUG_SYNC") == nullptr) {
        return;
    }
    const aclError ret = aclrtSynchronizeStream(stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "muon_polar_step: stage ", stage,
        " failed, ret=", ret, ", detail=", aclGetRecentErrMsg());
}

struct LaunchState {
    void *tilingDevice = nullptr;
    uint32_t blockDim = 0;

    ~LaunchState()
    {
        if (tilingDevice != nullptr) {
            (void)aclrtFree(tilingDevice);
        }
    }
};

struct GramLowLevelFusedCacheKey {
    int64_t deviceIndex;
    uintptr_t stream;
    uint32_t batch;
    uint32_t rows;
    uint32_t common;
    uint32_t coreDim;
    uint32_t groupTiles;

    bool operator==(const GramLowLevelFusedCacheKey &other) const
    {
        return std::tie(deviceIndex, stream, batch, rows, common, coreDim,
                groupTiles) ==
            std::tie(other.deviceIndex, other.stream, other.batch, other.rows,
                other.common, other.coreDim, other.groupTiles);
    }
};

struct GramLowLevelFusedCacheKeyHash {
    size_t operator()(const GramLowLevelFusedCacheKey &key) const
    {
        size_t h = std::hash<int64_t>{}(key.deviceIndex);
        const auto mix = [&h](auto value) {
            h ^= std::hash<decltype(value)>{}(value) +
                0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(key.stream);
        mix(key.batch);
        mix(key.rows);
        mix(key.common);
        mix(key.coreDim);
        mix(key.groupTiles);
        return h;
    }
};

std::mutex gGramLowLevelFusedCacheMutex;
std::unordered_map<GramLowLevelFusedCacheKey,
    std::shared_ptr<LaunchState>, GramLowLevelFusedCacheKeyHash>
    gGramLowLevelFusedCache;

uint16_t FloatToBf16Bits(float value)
{
    uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    const uint32_t rounded = raw + 0x7FFFU + ((raw >> 16U) & 1U);
    return static_cast<uint16_t>(rounded >> 16U);
}

uint32_t FloatRawBits(float value)
{
    uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

struct Cube2PCacheKey {
    int64_t deviceIndex;
    uintptr_t stream;
    uint32_t batch;
    uint32_t rows;
    uint32_t tile;
    uint32_t matmulDim;
    uint32_t bBits;
    uint32_t cBits;
    uint32_t persistentCoreDim;
    uint32_t panelM;

    bool operator==(const Cube2PCacheKey &other) const
    {
        return std::tie(deviceIndex, stream, batch, rows, tile, matmulDim,
            bBits, cBits, persistentCoreDim, panelM) ==
            std::tie(other.deviceIndex, other.stream, other.batch, other.rows,
                other.tile, other.matmulDim, other.bBits, other.cBits,
                other.persistentCoreDim, other.panelM);
    }
};

struct Cube2PCacheKeyHash {
    size_t operator()(const Cube2PCacheKey &key) const
    {
        size_t h = std::hash<int64_t>{}(key.deviceIndex);
        const auto mix = [&h](auto value) {
            h ^= std::hash<decltype(value)>{}(value) + 0x9e3779b97f4a7c15ULL +
                (h << 6) + (h >> 2);
        };
        mix(key.stream);
        mix(key.batch);
        mix(key.rows);
        mix(key.tile);
        mix(key.matmulDim);
        mix(key.bBits);
        mix(key.cBits);
        mix(key.persistentCoreDim);
        mix(key.panelM);
        return h;
    }
};

std::mutex gCube2PCacheMutex;
std::unordered_map<Cube2PCacheKey, std::shared_ptr<LaunchState>,
    Cube2PCacheKeyHash> gCube2PCache;

struct Cube3CacheKey {
    int64_t deviceIndex;
    uintptr_t stream;
    uint32_t batch;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t tile;
    uint32_t matmulDim;
    uint16_t alphaBits;
    bool bFullLoad;
    uint16_t panelM;

    bool operator==(const Cube3CacheKey &other) const
    {
        return std::tie(deviceIndex, stream, batch, m, n, k, tile, matmulDim,
            alphaBits, bFullLoad, panelM) ==
            std::tie(other.deviceIndex, other.stream, other.batch, other.m, other.n, other.k,
                other.tile, other.matmulDim, other.alphaBits,
                other.bFullLoad, other.panelM);
    }
};

struct Cube3CacheKeyHash {
    size_t operator()(const Cube3CacheKey &key) const
    {
        size_t h = std::hash<int64_t>{}(key.deviceIndex);
        const auto mix = [&h](auto value) {
            h ^= std::hash<decltype(value)>{}(value) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(key.stream);
        mix(key.batch);
        mix(key.m);
        mix(key.n);
        mix(key.k);
        mix(key.tile);
        mix(key.matmulDim);
        mix(key.alphaBits);
        mix(key.bFullLoad);
        mix(key.panelM);
        return h;
    }
};

std::mutex gCube3CacheMutex;
std::unordered_map<Cube3CacheKey, std::shared_ptr<LaunchState>,
    Cube3CacheKeyHash> gCube3Cache;

std::shared_ptr<LaunchState> GetGramLowLevelFusedLaunchState(
    const at::Tensor &x, uint32_t rows, uint32_t common, aclrtStream stream,
    uint32_t groupTiles)
{
    constexpr uint32_t tile = 128U;
    const uint32_t batch = static_cast<uint32_t>(x.size(0));
    const uint32_t coreDim = static_cast<uint32_t>(
        EnvU64("MUON_POLAR_CUBE1_LOWLEVEL_DIM", 28));
    TORCH_CHECK(coreDim > 0U && coreDim <= 128U,
        "muon_polar_step: MUON_POLAR_CUBE1_LOWLEVEL_DIM must be in [1, 128]");
    TORCH_CHECK(groupTiles == 1U || groupTiles == 2U,
        "muon_polar_step: Cube1 fused groupTiles must be 1 or 2");

    const GramLowLevelFusedCacheKey key{x.device().index(),
        reinterpret_cast<uintptr_t>(stream), batch, rows, common, coreDim,
        groupTiles};
    std::lock_guard<std::mutex> lock(gGramLowLevelFusedCacheMutex);
    const auto found = gGramLowLevelFusedCache.find(key);
    if (found != gGramLowLevelFusedCache.end()) {
        return found->second;
    }

    MuonGramLowLevelFusedTilingData hostTiling{};
    hostTiling.batch = batch;
    hostTiling.rows = rows;
    hostTiling.common = common;
    hostTiling.dim = rows / tile;
    hostTiling.groupTiles = groupTiles;
    for (uint32_t row = 0; row < hostTiling.dim; ++row) {
        const uint32_t tilesInRow = hostTiling.dim - row;
        hostTiling.groupsPerBatch +=
            (tilesInRow + groupTiles - 1U) / groupTiles;
    }
    const uint32_t totalWork = batch * hostTiling.groupsPerBatch;
    hostTiling.usedCoreNum = std::min(coreDim, totalWork);

    constexpr uint32_t transposeType = 7U;
    const auto makeTransposeTiling = [&](const ge::Shape &shape,
                                         ConfusionTransposeTiling &tiling) {
        uint32_t maxTmp = 0;
        uint32_t minTmp = 0;
        AscendC::GetConfusionTransposeMaxMinTmpSize(shape,
            sizeof(uint16_t), transposeType, maxTmp, minTmp);
        TORCH_CHECK(maxTmp <= 20U * 1024U,
            "muon_polar_step: Cube1 fused transpose tmp exceeds 20 KiB, got ",
            maxTmp);
        AscendC::GetConfusionTransposeOnlyTilingInfo(shape, maxTmp,
            sizeof(uint16_t), tiling);
    };
    makeTransposeTiling(ge::Shape({64, 128}),
        hostTiling.transposeTiling128);
    makeTransposeTiling(ge::Shape({64, 256}),
        hostTiling.transposeTiling256);

    auto state = std::make_shared<LaunchState>();
    state->blockDim = hostTiling.usedCoreNum;
    CheckAcl(aclrtMalloc(&state->tilingDevice, sizeof(hostTiling),
        ACL_MEM_MALLOC_HUGE_FIRST),
        "aclrtMalloc(gram low-level fused tiling)");
    CheckAcl(aclrtMemcpy(state->tilingDevice, sizeof(hostTiling), &hostTiling,
        sizeof(hostTiling), ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy(gram low-level fused tiling)");
    gGramLowLevelFusedCache.emplace(key, state);
    return state;
}

void LaunchGramLowLevelFused(const at::Tensor &input, at::Tensor &output,
    uint32_t rows, uint32_t common, aclrtStream stream, uint32_t groupTiles)
{
    auto state = GetGramLowLevelFusedLaunchState(input, rows, common, stream,
        groupTiles);
    launch_muon_gram_lowlevel_fused(input.data_ptr(), output.data_ptr(),
        state->tilingDevice, state->blockDim, stream);
}

std::shared_ptr<LaunchState> GetCube2PLaunchState(const at::Tensor &a,
    uint32_t rows, double b, double c, aclrtStream stream)
{
    const uint32_t batch = static_cast<uint32_t>(a.size(0));
    const uint32_t tile = static_cast<uint32_t>(EnvU64("MUON_POLAR_TILE", 128));
    const uint32_t matmulDim = static_cast<uint32_t>(
        EnvU64("MUON_POLAR_MATMUL_DIM", 80));
    const uint32_t bBits = FloatRawBits(static_cast<float>(b));
    const uint32_t cBits = FloatRawBits(static_cast<float>(c));
    const uint32_t persistentCoreDim = static_cast<uint32_t>(
        EnvU64("MUON_POLAR_CUBE2_PERSISTENT_DIM", 40));
    const uint32_t estimatedDim = rows / 128U;
    const char *panelMEnv = std::getenv("MUON_POLAR_CUBE2_PANEL_M");
    // Cube2 只调度上三角，列长度不均衡；9.0 实测最佳 panel 与 Cube3
    // 不同。d12/d16(dim=6/8) 取 6，其余正式/单 batch 取 2
    // 以保留足够 job 并减轻长列尾部负载不均。
    const uint32_t autoPanelM = batch > 1U &&
            (estimatedDim == 6U || estimatedDim == 8U)
        ? 6U
        : std::min(2U, estimatedDim);
    const uint32_t panelM = panelMEnv != nullptr
        ? static_cast<uint32_t>(EnvU64("MUON_POLAR_CUBE2_PANEL_M", 1))
        : autoPanelM;

    TORCH_CHECK(tile == 128,
        "muon_polar_step: Cube2+P fusion currently requires MUON_POLAR_TILE=128");
    TORCH_CHECK(rows % 128U == 0U,
        "muon_polar_step: Cube2+P fusion currently requires rows to be 128-aligned");
    TORCH_CHECK(matmulDim > 0 && matmulDim <= 80,
        "muon_polar_step: Cube2+P fusion requires MUON_POLAR_MATMUL_DIM in [1, 80]");
    TORCH_CHECK(persistentCoreDim > 0U && persistentCoreDim <= 40U,
        "muon_polar_step: MUON_POLAR_CUBE2_PERSISTENT_DIM must be in [1, 40]");
    TORCH_CHECK(panelM > 0U && panelM <= estimatedDim,
        "muon_polar_step: MUON_POLAR_CUBE2_PANEL_M must be in [1, rows/128]");

    const Cube2PCacheKey key{a.device().index(),
        reinterpret_cast<uintptr_t>(stream), batch, rows, tile, matmulDim,
        bBits, cBits, persistentCoreDim, panelM};
    std::lock_guard<std::mutex> lock(gCube2PCacheMutex);
    const auto found = gCube2PCache.find(key);
    if (found != gCube2PCache.end()) {
        return found->second;
    }

    matmul_tiling::MultiCoreMatmulTiling mmTiling;
    mmTiling.SetAType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16,
        false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16,
        true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetShape(static_cast<int32_t>(rows), static_cast<int32_t>(rows),
        static_cast<int32_t>(rows));
    mmTiling.SetOrgShape(static_cast<int32_t>(rows), static_cast<int32_t>(rows),
        static_cast<int32_t>(rows));
    mmTiling.SetBias(false);
    TORCH_CHECK(mmTiling.SetSingleShape(static_cast<int32_t>(tile),
        static_cast<int32_t>(tile), -1) == 0,
        "muon_polar_step: Cube2+P SetSingleShape failed");
    mmTiling.SetDim(static_cast<int32_t>(matmulDim));
    mmTiling.EnableMultiCoreSplitK(false);

    MuonCube2PTilingData hostTiling{};
    hostTiling.batch = batch;
    hostTiling.rows = rows;
    hostTiling.bBits = bBits;
    hostTiling.cBits = cBits;
    hostTiling.reserved = static_cast<uint32_t>(panelM << 8U);
    TORCH_CHECK(mmTiling.GetTiling(hostTiling.cubeTiling) == 0,
        "muon_polar_step: Cube2+P MultiCoreMatmulTiling failed");

    const ge::Shape transposeShape({64, 128});
    // TransposeType::TRANSPOSE_ND2ND_ONLY 在 CANN 9.0 中的枚举值为 7。
    constexpr uint32_t transposeType = 7U;
    uint32_t transposeMaxTmp = 0;
    uint32_t transposeMinTmp = 0;
    AscendC::GetConfusionTransposeMaxMinTmpSize(transposeShape,
        sizeof(uint16_t), transposeType, transposeMaxTmp, transposeMinTmp);
    AscendC::GetConfusionTransposeOnlyTilingInfo(transposeShape,
        transposeMaxTmp, sizeof(uint16_t), hostTiling.transposeTiling);

    int32_t rawBlockDim = 0;
    int32_t mDim = 0;
    int32_t nDim = 0;
    TORCH_CHECK(mmTiling.GetCoreNum(rawBlockDim, mDim, nDim) == 0,
        "muon_polar_step: Cube2+P GetCoreNum failed");
    TORCH_CHECK(rawBlockDim > 0 && mDim > 0 && mDim == nDim,
        "muon_polar_step: Cube2+P invalid square tiling");
    TORCH_CHECK(hostTiling.cubeTiling.singleCoreM == 128 &&
        hostTiling.cubeTiling.singleCoreN == 128,
        "muon_polar_step: Cube2+P mixed kernel requires 128x128 Cube tiles");
    hostTiling.mDim = static_cast<uint32_t>(mDim);
    hostTiling.nDim = static_cast<uint32_t>(nDim);
    uint32_t jobsPerBatch = 0;
    for (uint32_t tileN = 0; tileN < hostTiling.nDim; ++tileN) {
        jobsPerBatch += (tileN + 1U + panelM - 1U) / panelM;
    }
    const uint32_t blockDim =
        std::min(persistentCoreDim, batch * jobsPerBatch);
    hostTiling.usedCoreNum = blockDim;

    auto state = std::make_shared<LaunchState>();
    state->blockDim = blockDim;
    CheckAcl(aclrtMalloc(&state->tilingDevice, sizeof(hostTiling),
        ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(cube2_p tiling)");
    CheckAcl(aclrtMemcpy(state->tilingDevice, sizeof(hostTiling), &hostTiling,
        sizeof(hostTiling), ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy(cube2_p tiling)");
    gCube2PCache.emplace(key, state);
    return state;
}

void LaunchCube2P(const at::Tensor &a, at::Tensor &p, uint32_t rows,
    double b, double c, aclrtStream stream)
{
    auto state = GetCube2PLaunchState(a, rows, b, c, stream);
    launch_muon_cube2_p_fused(a.data_ptr(), p.data_ptr(),
        state->tilingDevice, state->blockDim, stream);
}

std::shared_ptr<LaunchState> GetCube3LaunchState(const at::Tensor &polynomial,
    uint32_t m, uint32_t n, uint32_t k, double alpha, aclrtStream stream)
{
    const uint32_t batch = static_cast<uint32_t>(polynomial.size(0));
    const uint32_t tile = static_cast<uint32_t>(EnvU64("MUON_POLAR_CUBE3_TILE", 128));
    // mixed kernel 是 1AIC:2AIV，SetDim 以 AIC 数量配置。
    const uint32_t matmulDim = static_cast<uint32_t>(EnvU64("MUON_POLAR_CUBE3_DIM", 40));
    const uint16_t alphaBits = FloatToBf16Bits(static_cast<float>(alpha));
    const uint32_t estimatedMDim = (m + 127U) / 128U;
    const uint32_t estimatedNDim = (n + 127U) / 128U;
    const uint64_t alignedK = (static_cast<uint64_t>(k) + 15ULL) & ~15ULL;
    constexpr uint64_t autoAL1Bytes = 128ULL * 256ULL * 2ULL * 2ULL;
    const uint64_t autoBL1Bytes = 128ULL * alignedK * 2ULL;
    // 9.0/dav-c310 实测：多 batch 下 MDim<=8 时完整复用一份 B panel
    // 最快；MDim=12 时 panelM=6 在并行度与复用之间更稳定。单 batch
    // 需要更多 job 覆盖 AIC，panelM=2 明显优于原来的 4。
    const uint32_t autoPanelM = batch == 1U
        ? std::min(2U, estimatedMDim)
        : (estimatedMDim <= 8U ? estimatedMDim
                               : std::min(6U, estimatedMDim));
    const uint32_t autoMGroups =
        (estimatedMDim + autoPanelM - 1U) / autoPanelM;
    const uint64_t autoJobs = static_cast<uint64_t>(batch) * estimatedNDim *
        autoMGroups;
    const bool autoBFullLoad = estimatedMDim >= 4U &&
        autoAL1Bytes + autoBL1Bytes <= 512ULL * 1024ULL &&
        autoJobs >= std::min(matmulDim, 40U);
    const char *bFullLoadEnv = std::getenv("MUON_POLAR_CUBE3_B_FULL_LOAD");
    const bool bFullLoad = bFullLoadEnv != nullptr
        ? EnvU64("MUON_POLAR_CUBE3_B_FULL_LOAD", 0) != 0
        : autoBFullLoad;
    const char *panelMEnv = std::getenv("MUON_POLAR_CUBE3_PANEL_M");
    const uint32_t panelM = bFullLoad
        ? (panelMEnv != nullptr
            ? static_cast<uint32_t>(EnvU64("MUON_POLAR_CUBE3_PANEL_M", 4))
            : autoPanelM)
        : 1U;
    TORCH_CHECK(tile == 128,
        "muon_polar_step: experimental fused Cube3 currently requires MUON_POLAR_CUBE3_TILE=128");
    TORCH_CHECK(matmulDim > 0 && matmulDim <= 40,
        "muon_polar_step: MUON_POLAR_CUBE3_DIM must be in [1, 40]");
    TORCH_CHECK(m % 16 == 0 && n % 16 == 0 && k % 16 == 0,
        "muon_polar_step: fused Cube3 currently requires M/N/K to be 16-aligned");
    TORCH_CHECK(panelM > 0 && panelM <= 255,
        "muon_polar_step: MUON_POLAR_CUBE3_PANEL_M must be in [1, 255]");

    const Cube3CacheKey key{polynomial.device().index(), reinterpret_cast<uintptr_t>(stream),
        batch, m, n, k, tile, matmulDim, alphaBits, bFullLoad,
        static_cast<uint16_t>(panelM)};
    std::lock_guard<std::mutex> lock(gCube3CacheMutex);
    const auto found = gCube3Cache.find(key);
    if (found != gCube3Cache.end()) {
        return found->second;
    }

    matmul_tiling::MultiCoreMatmulTiling mmTiling;
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BF16, false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BF16, false);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    mmTiling.SetOrgShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    mmTiling.SetBias(false);
    TORCH_CHECK(mmTiling.SetSingleShape(static_cast<int32_t>(tile), static_cast<int32_t>(tile), -1) == 0,
        "muon_polar_step: fused Cube3 SetSingleShape failed");
    mmTiling.SetDim(static_cast<int32_t>(matmulDim));
    mmTiling.EnableMultiCoreSplitK(false);

    MuonCube3TilingData hostTiling{};
    hostTiling.batch = batch;
    hostTiling.m = m;
    hostTiling.n = n;
    hostTiling.k = k;
    hostTiling.alphaBits = alphaBits;
    hostTiling.reserved = 0U;
    if (bFullLoad) {
        // bit 1 选择 B 全载；高 8 bit 保存一个 B panel 复用的 tileM 数。
        hostTiling.reserved |= 2U;
        hostTiling.reserved |= static_cast<uint16_t>(panelM << 8U);
    }
    TORCH_CHECK(mmTiling.GetTiling(hostTiling.cubeTiling) == 0,
        "muon_polar_step: fused Cube3 MultiCoreMatmulTiling failed");
    if (bFullLoad) {
        // 3510 L1 共 512 KiB。当前 BlockMmad 配置为两份 128x256 AL1
        // 缓冲，随后放置一份 singleCoreN x alignedK 的完整 BL1 panel。
        constexpr uint64_t kL1Bytes = 512ULL * 1024ULL;
        constexpr uint64_t kL1TileK = 256ULL;
        constexpr uint64_t kL1BufferCount = 2ULL;
        constexpr uint64_t kBf16Bytes = 2ULL;
        const uint64_t alignedK = (static_cast<uint64_t>(k) + 15ULL) & ~15ULL;
        const uint64_t aL1Bytes =
            static_cast<uint64_t>(hostTiling.cubeTiling.singleCoreM) *
            kL1TileK * kL1BufferCount * kBf16Bytes;
        const uint64_t bL1Bytes =
            static_cast<uint64_t>(hostTiling.cubeTiling.singleCoreN) *
            alignedK * kBf16Bytes;
        TORCH_CHECK(aL1Bytes + bL1Bytes <= kL1Bytes,
            "muon_polar_step: Cube3 B full-load needs ", aL1Bytes + bL1Bytes,
            " bytes of L1, exceeding the 3510 limit of ", kL1Bytes,
            " bytes; disable MUON_POLAR_CUBE3_B_FULL_LOAD for this shape");
    }

    int32_t rawBlockDim = 0;
    int32_t mDim = 0;
    int32_t nDim = 0;
    TORCH_CHECK(mmTiling.GetCoreNum(rawBlockDim, mDim, nDim) == 0,
        "muon_polar_step: fused Cube3 GetCoreNum failed");
    TORCH_CHECK(rawBlockDim > 0 && mDim > 0 && nDim > 0,
        "muon_polar_step: fused Cube3 invalid tiling");
    hostTiling.mDim = static_cast<uint32_t>(mDim);
    hostTiling.nDim = static_cast<uint32_t>(nDim);
    const uint32_t tilesPerBatch = hostTiling.mDim * hostTiling.nDim;
    uint32_t blockDim = 0;
    if (bFullLoad) {
        // 一个 job 先把固定 tileN 的完整 KxN panel 搬入 BL1，再连续
        // 计算 panelM 个 tileM，使 GM->L1 的 X 搬运得到复用。
        const uint32_t mGroupCount =
            (hostTiling.mDim + panelM - 1U) / panelM;
        const uint32_t totalJobs = batch * hostTiling.nDim * mGroupCount;
        blockDim = std::min(static_cast<uint32_t>(rawBlockDim), totalJobs);
    } else {
        // 低层 BlockMmad 自己管理 L0C/Fixpipe 双缓冲，可由全局 40 个
        // 常驻 AIC block 循环覆盖所有 batch/tile。
        blockDim = std::min(static_cast<uint32_t>(rawBlockDim), batch * tilesPerBatch);
    }
    hostTiling.usedCoreNum = blockDim;

    auto state = std::make_shared<LaunchState>();
    state->blockDim = blockDim;
    CheckAcl(aclrtMalloc(&state->tilingDevice, sizeof(hostTiling), ACL_MEM_MALLOC_HUGE_FIRST),
        "aclrtMalloc(cube3 tiling)");
    CheckAcl(aclrtMemcpy(state->tilingDevice, sizeof(hostTiling), &hostTiling, sizeof(hostTiling),
        ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(cube3 tiling)");
    gCube3Cache.emplace(key, state);
    return state;
}

void LaunchCube3Fused(const at::Tensor &polynomial, const at::Tensor &x, at::Tensor &out,
    double alpha, aclrtStream stream)
{
    const uint32_t m = static_cast<uint32_t>(polynomial.size(1));
    const uint32_t k = static_cast<uint32_t>(polynomial.size(2));
    const uint32_t n = static_cast<uint32_t>(x.size(2));
    auto state = GetCube3LaunchState(polynomial, m, n, k, alpha, stream);
    launch_muon_cube3_fused_lowlevel(polynomial.data_ptr(), x.data_ptr(),
        out.data_ptr(), state->tilingDevice, state->blockDim, stream);
}

at::Tensor MuonPolarStepAscend950(const at::Tensor &x, double a,
    double b, double c)
{
    TORCH_CHECK(x.defined(), "muon_polar_step: x must be defined");
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
        "muon_polar_step: x must be an NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "muon_polar_step: only bfloat16 is supported");
    TORCH_CHECK(x.dim() == 3, "muon_polar_step: x must have shape [batch, M, N]");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
        "muon_polar_step: all dimensions must be positive");
    TORCH_CHECK(x.size(1) <= x.size(2),
        "muon_polar_step: current prototype expects M <= N; transpose policy stays outside this op");
    const c10::OptionalDeviceGuard deviceGuard(x.device());
    at::Tensor input = x.contiguous();
    const int64_t batch = input.size(0);
    const int64_t originalRows = input.size(1);
    const int64_t originalCols = input.size(2);
    const int64_t rows = AlignUp(originalRows, 128);
    const int64_t cols = std::max(AlignUp(originalCols, 16), rows);
    const bool padded = rows != originalRows || cols != originalCols;
    if (padded) {
        at::Tensor paddedInput = at::zeros({batch, rows, cols},
            input.options().memory_format(c10::MemoryFormat::Contiguous));
        paddedInput.narrow(1, 0, originalRows)
            .narrow(2, 0, originalCols)
            .copy_(input);
        input = paddedInput;
    }
    auto squareOptions = input.options().memory_format(c10::MemoryFormat::Contiguous);
    // raw ACL kernel launch 之前必须先排空 torch_npu 的 host task queue，
    // 否则 Cube 可能早于输入的异步拷贝/缩放执行。这里只 flush，不做设备同步。
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    at::Tensor gram = at::empty({batch, rows, rows}, squareOptions);
    at::Tensor polynomial;
    // Cube1：X 与自身共享同一份 GM，B 端仅设置 transpose 标志。
    const uint32_t effectiveGroupTiles = rows <= 512 ? 1U : 2U;
    LaunchGramLowLevelFused(input, gram,
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(cols), stream,
        effectiveGroupTiles);
    DebugSync(stream, "cube1_gram_lowlevel_fused");

    // Cube2 的 FP32 L0C 经 dual Fixpipe 进入两个 AIV，在 UB 中直接生成 P。
    polynomial = at::empty({batch, rows, rows}, squareOptions);
    LaunchCube2P(gram, polynomial, static_cast<uint32_t>(rows), b, c,
        stream);
    DebugSync(stream, "cube2_p_fused");

    // Cube3 的 L0C 经 dual Fixpipe 进入两个 AIV，在 UB 中直接生成 Y。
    at::Tensor out = at::empty_like(input,
        input.options().memory_format(c10::MemoryFormat::Contiguous));
    LaunchCube3Fused(polynomial, input, out, a, stream);
    DebugSync(stream, "cube3_aiv_fused");
    if (!padded) {
        return out;
    }
    return out.narrow(1, 0, originalRows)
        .narrow(2, 0, originalCols)
        .contiguous();
}

}  // namespace

TORCH_LIBRARY(muon_ascend950, m)
{
    m.def("polar_step(Tensor x, float a, float b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(muon_ascend950, PrivateUse1, m)
{
    m.impl("polar_step", &MuonPolarStepAscend950);
}
