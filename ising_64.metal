#include <metal_stdlib>
using namespace metal;

// Simulation parameters struct
struct SimParams {
    uint32_t L;      // Lattice size
    int J;           // Coupling constant
    float T;         // Temperature
    uint32_t seed;   // Random seed
};

// Structure for storing 64 parallel spins
struct SpinBlock {
    uint64_t spins;  // Each bit represents one simulation
};

// PCG Random Number Generator
class PCGState {
private:
    uint64_t state;
    uint64_t inc;
    
public:
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
        return float(next()) / float(0xFFFFFFFFu);
    }
};

// Helper function for periodic boundary conditions
inline int periodic(uint32_t i, uint32_t limit) {
    return int(i % limit);
}

// Initialize spins for all 64 parallel simulations
kernel void initialize_spins(device SpinBlock* spin_blocks [[buffer(0)]],
                           constant SimParams& params [[buffer(1)]],
                           uint2 pos [[thread_position_in_grid]]) {
    // Boundary check
    if (pos.x >= params.L || pos.y >= params.L) return;
    
    uint32_t idx = pos.y * params.L + pos.x;
    
    // Initialize RNG with unique seed for this position
    PCGState rng(params.seed + idx);
    
    // Initialize all 64 parallel simulations
    uint64_t block = 0;
    for (int sim = 0; sim < 64; ++sim) {
        // 75% probability of +1 spin for faster equilibration
        if (rng.nextFloat() < 0.75f) {
            block |= (1ULL << sim);
        }
    }
    
    spin_blocks[idx].spins = block;
}

// Update spins using checkerboard decomposition
kernel void update_spins(device SpinBlock* spin_blocks [[buffer(0)]],
                        constant SimParams& params [[buffer(1)]],
                        constant uint32_t& color [[buffer(2)]],
                        constant float& prob_pos4 [[buffer(3)]],
                        constant float& prob_pos8 [[buffer(4)]],
                        uint2 pos [[thread_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    // Boundary and checkerboard pattern check
    if (pos.x >= params.L || pos.y >= params.L) return;
    if (((pos.x + pos.y) % 2) != color) return;

    uint32_t x = pos.x;
    uint32_t y = pos.y;
    uint32_t idx = x * params.L + y;

    // Get center and neighboring spins with periodic boundary conditions
    uint64_t center = spin_blocks[idx].spins;
    uint64_t up = spin_blocks[periodic(x-1, params.L) * params.L + y].spins;
    uint64_t down = spin_blocks[periodic(x+1, params.L) * params.L + y].spins;
    uint64_t left = spin_blocks[x * params.L + periodic(y-1, params.L)].spins;
    uint64_t right = spin_blocks[x * params.L + periodic(y+1, params.L)].spins;

    // Calculate aligned neighbors using bitwise operations
    uint64_t aligned_up = ~(center ^ up);
    uint64_t aligned_down = ~(center ^ down);
    uint64_t aligned_left = ~(center ^ left);
    uint64_t aligned_right = ~(center ^ right);

    uint64_t ud_sum = aligned_up ^ aligned_down; 
    uint64_t ud_c = aligned_up & aligned_down;

    uint64_t lr_sum = aligned_left ^ aligned_right; 
    uint64_t lr_c = aligned_left & aligned_right;

    uint64_t b0 = ud_sum ^ lr_sum; 
    uint64_t udlr_c = ud_sum & lr_sum;

    uint64_t b1 = udlr_c ^ (ud_c ^ lr_c);
    uint64_t b2 = (ud_c & lr_c) | (lr_c & udlr_c) | (ud_c & udlr_c);

    // Create masks for each energy level
    uint64_t de_neg8 = ~b0 & ~b1 & ~b2;
    uint64_t de_neg4 = b0 & ~b1 & ~b2;
    uint64_t de_zero = ~b0 & b1 & ~b2;
    uint64_t de_pos4 = b0 & b1 & ~b2;
    uint64_t de_pos8 = ~b0 & ~b1 & b2;

    // Initialize RNG for acceptance probabilities
    PCGState rng(params.seed + idx + tid);

    // Create flip mask based on Metropolis acceptance probabilities
    uint64_t flip_mask = de_neg8 | de_neg4 | de_zero; // Always accept negative energy changes

    float random_float = rng.nextFloat();
    if (random_float < prob_pos4) {
        flip_mask |= de_pos4;
    }
    if (random_float < prob_pos8) {
        flip_mask |= de_pos8;
    }

    // Apply accepted spin flips
    spin_blocks[idx].spins ^= flip_mask;
}


// Calculate magnetization for all simulations
kernel void calculate_magnetization(device const SpinBlock* spin_blocks [[buffer(0)]],
                                  device atomic_int* magnetization [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]],
                                  uint grid_size [[threads_per_grid]]) {
    if (tid >= grid_size) return;
    
    uint64_t block = spin_blocks[tid].spins;
    
    // Process each simulation independently
    for (int sim = 0; sim < 64; sim++) {
        // Extract spin value for this simulation
        int spin_value = ((block >> sim) & 1ULL) ? 1 : -1;
        // Add to the corresponding magnetization counter
        atomic_fetch_add_explicit(&magnetization[sim], spin_value, memory_order_relaxed);
    }
}