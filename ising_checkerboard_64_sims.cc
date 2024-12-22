#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <functional>
#include <inttypes.h>

#include "DispatchQueue.cc"

// Cache line size (typical for most modern processors)
constexpr int CACHE_LINE_SIZE = 64;

const int L = 20000;           // Lattice size
const int N = L * L;         // Number of spins per simulation
const int NUM_PARALLEL_SIMS = 64;  // Number of parallel simulations
const int steps = 10;       // Number of Monte Carlo steps 
const int J = 1;            // Interaction strength
const double T = 2.0;        // Temperature (in units of J/k_B)

const int NUM_THREADS = std::thread::hardware_concurrency();
const bool PRINT_SPINS = false;
const bool PRINT_MAGNETIZATION = true;

constexpr int MAX_DE = 8 * J;
alignas(CACHE_LINE_SIZE) std::vector<double> exp_lookup(2 * MAX_DE + 1);

// Cache-friendly data structure for spin blocks
struct alignas(CACHE_LINE_SIZE) SpinBlock {
    uint64_t spins;  // Each bit represents the same position in different simulations
    uint64_t padding[7];
};

// Thread-local RNG
struct Xorshiro {
    uint64_t state[2];

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    Xorshiro(uint64_t seed) {
        uint64_t z = seed;
        // Splitmix64 initialization
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        state[0] = z;

        z = seed + 0x9E3779B97F4A7C15ULL; // Another good seed value
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        state[1] = z;
    }

    uint64_t next() {
        const uint64_t s0 = state[0];
        uint64_t s1 = state[1];
        const uint64_t result = rotl(s0 * 5, 7) * 9;

        s1 ^= s0;
        state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        state[1] = rotl(s1, 37);

        return result;
    }

    double next_double() {
        return static_cast<double>(next() >> 11) / (1ULL << 53);
    }
};

thread_local Xorshiro rng(std::random_device{}());

void initialize_exp_lookup(double T) {
    for (int dE = -MAX_DE; dE <= MAX_DE; ++dE) {
        exp_lookup[dE + MAX_DE] = exp(-dE / T);
    }
}

// Initialize spins for all parallel simulations with 75% probability of +1
void initialize_spins(std::vector<SpinBlock> &spin_blocks) {
    const double prob = 0.75;  // Probability of +1 spin
    
    for (int i = 0; i < N; ++i) {
        uint64_t block = 0;
        
        // Generate 64 random values and set bits based on probability
        for (int bit = 0; bit < 64; ++bit) {
            if (rng.next() % 100 < prob * 100) {  // Using integer arithmetic for speed
                block |= (1ULL << bit);
            }
        }
        
        spin_blocks[i].spins = block;
    }
}

// Helper function to get neighbor spins for all simulations
inline uint64_t get_neighbor_spins(const std::vector<SpinBlock>& spin_blocks, int idx) {
    return spin_blocks[idx].spins;
}

// Periodic boundary conditions
inline int periodic(int i, int limit) {
    return (i + limit) % limit;
}

// Calculate energy changes for all simulations simultaneously using bit operations
std::vector<uint64_t> delta_energy(const std::vector<SpinBlock> &spins, int x, int y) {
    uint64_t center = get_neighbor_spins(spins, x * L + y);
    uint64_t up = get_neighbor_spins(spins, periodic(x - 1, L) * L + y);
    uint64_t down = get_neighbor_spins(spins, periodic(x + 1, L) * L + y);
    uint64_t left = get_neighbor_spins(spins, x * L + periodic(y - 1, L));
    uint64_t right = get_neighbor_spins(spins, x * L + periodic(y + 1, L));    
    
    // Bitwise adding 
    uint64_t aligned_up = ~(center ^ up);
    uint64_t aligned_down = ~(center ^ down);
    uint64_t aligned_left = ~(center ^ left);
    uint64_t aligned_right = ~(center ^ right);

    uint64_t ud_sum = (aligned_up ^ aligned_down); //1st sum
    uint64_t ud_c = (aligned_up & aligned_down); //1st carry

    uint64_t lr_sum = (aligned_left ^ aligned_right); //1st sum
    uint64_t lr_c= (aligned_left & aligned_right); //2nd carry

    //add sums
    uint64_t b0 = (ud_sum ^ lr_sum); 
    uint64_t udlr_c = (ud_sum & lr_sum); //3rd carry

    //Full adder formula for carries
    uint64_t b1 = udlr_c ^ (ud_c ^ lr_c);
    uint64_t b2 = (ud_c & lr_c) | (lr_c & udlr_c) | (ud_c & udlr_c);

    // Create masks for each energy level
    uint64_t de_neg8 = ~b0 & ~b1 & ~b2;                // 0 aligned
    uint64_t de_neg4 = b0 & ~b1 & ~b2;                 // 1 aligned
    uint64_t de_zero = ~b0 & b1 & ~b2;                 // 2 aligned
    uint64_t de_pos4 = b0 & b1 & ~b2;                  // 3 aligned
    uint64_t de_pos8 = ~b0 & ~b1 & b2;                 // 4 aligned

    std::vector<uint64_t> result = {de_neg8,de_neg4,de_zero,de_pos4,de_pos8};

    return result;
}

// Monte Carlo step for all simulations simultaneously
void monte_carlo_step(std::vector<SpinBlock>& spin_blocks, int x, int y, Xorshiro& local_rng) {
    std::vector<uint64_t> dE = delta_energy(spin_blocks, x, y);
    
   // Generate random bits for acceptance probabilities
    double rand_bits = local_rng.next_double();
    // Always accept negative dE
    uint64_t flip_mask = dE[0] | dE[1];
    // Accept positive dE based on exp_lookup probabilities
    uint64_t accept_pos0 = dE[2] & ~(static_cast<uint64_t>(rand_bits < exp_lookup[MAX_DE]) - 1);
    uint64_t accept_pos4 = dE[3] & ~(static_cast<uint64_t>(rand_bits < exp_lookup[MAX_DE+4]) - 1);
    uint64_t accept_pos8 = dE[4] & ~(static_cast<uint64_t>(rand_bits < exp_lookup[MAX_DE+8]) - 1);

    flip_mask |= accept_pos0 | accept_pos4 | accept_pos8;
    // Apply all accepted flips at once
    spin_blocks[x * L + y].spins ^= flip_mask;
}

// Calculate magnetization for all simulations
std::vector<double> magnetization(const std::vector<SpinBlock> &spins) {
    std::vector<int> counts(NUM_PARALLEL_SIMS, 0);
    
    for (int i = 0; i < N; ++i) {
        uint64_t block = spins[i].spins;
        for (int sim = 0; sim < NUM_PARALLEL_SIMS; ++sim) {
            counts[sim] += (block >> sim) & 1 ? 1 : -1;
        }
    }
    
    std::vector<double> mags(NUM_PARALLEL_SIMS);
    for (int sim = 0; sim < NUM_PARALLEL_SIMS; ++sim) {
        mags[sim] = static_cast<double>(counts[sim]) / N;
    }
    return mags;
}

// Update spins of one color for all simulations
void update_spins(std::vector<SpinBlock> &spins, int color, int num_threads, DispatchQueue* dispatch_queue) {
    int sites_per_thread = N / num_threads;

    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        dispatch_queue->dispatch([&,thread_id] (int) {
            Xorshiro& local_rng = rng;  // Get TLS reference once at the start of thread
            for (int idx = thread_id * sites_per_thread; idx < (thread_id + 1) * sites_per_thread; ++idx) {
                int x = idx / L;
                int y = idx % L;
                if ((x + y) % 2 == color) {
                    monte_carlo_step(spins, x, y, local_rng);
                }
            }
        });
    }
    dispatch_queue->finishTasks();
}

void run_simulation(int num_threads, std::ofstream &prof_file) {
    std::vector<SpinBlock> spins(N);
    DispatchQueue* dispatch_queue = new DispatchQueue((size_t) num_threads);

    initialize_exp_lookup(T);
    
    auto init_start_time = std::chrono::high_resolution_clock::now();
    initialize_spins(spins);
    auto init_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> init_duration = init_end_time - init_start_time;

    std::vector<double> mags;
    // Print initial magnetization for first simulation
    if(PRINT_MAGNETIZATION) {
        mags = magnetization(spins);
        prof_file << "Step 0 Magnetization (sim 0): " << mags[0] << "\n";
    }

    auto update_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 1; i <= steps; ++i) {
        update_spins(spins, 0, num_threads, dispatch_queue);
        update_spins(spins, 1, num_threads, dispatch_queue);

        if (i % 1 == 0) {
            if(PRINT_MAGNETIZATION) {
                mags = magnetization(spins);
                prof_file << "Step " << i << " Magnetization (sim 0): " << mags[0] << "\n";
            }
        }
    }

    auto update_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> update_duration = update_end_time - update_start_time;

    // Multiply by NUM_PARALLEL_SIMS since we're doing 64 simulations at once
    long long total_spin_flips = (long long)steps * N * NUM_PARALLEL_SIMS;
    prof_file << "\nNumber of Threads,Initialization Time (seconds),"
              << "Update Time (seconds),Spin Flips per Second\n";
    prof_file << num_threads << "," 
              << init_duration.count() << "," 
              << update_duration.count() << ","
              << total_spin_flips / update_duration.count() << "\n";


    std::cout << num_threads << "," 
              << init_duration.count() << "," 
              << update_duration.count() << ","
              << total_spin_flips / update_duration.count() << "\n";
              
    delete dispatch_queue;
}

int main() {
    std::cout << "Running parallel Ising model simulation (" 
              << NUM_PARALLEL_SIMS << " simultaneous simulations)...\n";
    
    std::ofstream prof_file("profiling_data_parallel.csv");

    for (int num_threads = 1; num_threads <= NUM_THREADS; ++num_threads) {
        run_simulation(num_threads, prof_file);
    }
    
    prof_file.close();
    return 0;
}