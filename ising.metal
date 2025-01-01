#include <metal_stdlib>
using namespace metal;

struct SimParams {
    uint32_t L;  // Changed to unsigned
    int J;
    float T;
    uint32_t seed;
};

struct PCGState {
    uint64_t state;
    uint64_t inc;
    
    PCGState(uint64_t initstate = 0x853c49e6748fea9bULL, uint64_t initseq = 0xda3e39cb94b95bdbULL) {
        state = 0;
        inc = (initseq << 1u) | 1u;
        next();
        state += initstate;
        next();
    }
    
    uint32_t next() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = uint32_t(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    
    float nextFloat() {
        return float(next()) / float(0xFFFFFFFFu);  // Using hex literal instead of UINT32_MAX
    }
};

kernel void initialize_spins(device int* spins [[buffer(0)]],
                           constant SimParams& params [[buffer(1)]],
                           uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= params.L || pos.y >= params.L) return;
    
    uint32_t idx = pos.y * params.L + pos.x;
    PCGState rng(params.seed + idx);
    spins[idx] = (rng.nextFloat() < 0.25f) ? -1 : 1;
}

inline int periodic(uint32_t i, uint32_t limit) {  // Changed to unsigned
    return int(i % limit);
}

kernel void update_spins(device int* spins [[buffer(0)]],
                        constant SimParams& params [[buffer(1)]],
                        uint2 pos [[thread_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        constant uint32_t& color [[buffer(2)]]) {  // Fixed buffer type
    if (pos.x >= params.L || pos.y >= params.L) return;
    if (((pos.x + pos.y) % 2) != color) return;
    
    uint32_t x = pos.x;
    uint32_t y = pos.y;
    uint32_t idx = x * params.L + y;
    
    int spin = spins[idx];
    int neighbors = 
        spins[periodic(x-1, params.L) * params.L + y] +
        spins[periodic(x+1, params.L) * params.L + y] +
        spins[x * params.L + periodic(y-1, params.L)] +
        spins[x * params.L + periodic(y+1, params.L)];
    
    int dE = 2 * params.J * spin * neighbors;
    
    PCGState rng(params.seed + idx + tid);
    if (dE <= 0 || rng.nextFloat() < exp(-dE / params.T)) {
        spins[idx] *= -1;
    }
}

kernel void calculate_magnetization(device const int* spins [[buffer(0)]],
                                  device atomic_int* magnetic_sum [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]],
                                  uint grid_size [[threads_per_grid]]) {
    if (tid >= grid_size) return;
    atomic_fetch_add_explicit(magnetic_sum, spins[tid], memory_order_relaxed);
}