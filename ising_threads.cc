#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <thread>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <functional>
#include <inttypes.h>

#include "DispatchQueue.cc"

const int L = 20000;      // Lattice size
const int N = L * L;     // Number of spins
const int steps = 10;   // Number of Monte Carlo steps
const int J = 1;         // Interaction strength
const double T = 2.0;    // Temperature (in units of J/k_B)

const int NUM_THREADS = std::thread::hardware_concurrency(); // Get the number of available threads

// Pre-compute exponentials for possible energy changes
constexpr int MAX_DE = 8 * J;
std::vector<double> exp_lookup(2 * MAX_DE + 1);
// Function to initialize the lookup table
void initialize_exp_lookup(double T) {
    for (int dE = -MAX_DE; dE <= MAX_DE; ++dE) {
        exp_lookup[dE + MAX_DE] = exp(-dE / T);
    }
}

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

// Thread-local RNG
thread_local Xorshiro rng(std::random_device{}());



// Function to initialize spins with more -1 values
void initialize_spins(std::vector<int>& spins) {
    for (int i = 0; i < N; ++i) {
        spins[i] = (rng.next_double() < 0.25) ? -1 : 1;
    }
}

// Periodic boundary conditions
inline int periodic(int i, int limit) {
    return (i + limit) % limit;
}

// Function to calculate energy change when flipping a spin
int delta_energy(const std::vector<int>& spins, int x, int y) {
    int spin = spins[x * L + y];
    int neighbors = 
        spins[periodic(x-1, L) * L + y] +
        spins[periodic(x+1, L) * L + y] +
        spins[x * L + periodic(y-1, L)] +
        spins[x * L + periodic(y+1, L)];
    
    return 2 * J * spin * neighbors;
}

// Function to perform one Monte Carlo step for a single spin
void monte_carlo_step(std::vector<int>& spins, int x, int y, int idx, Xorshiro& local_rng) {
    int dE = delta_energy(spins, x, y);
           
    // Use pre-computed exponentials
    if (dE <= 0 || local_rng.next_double() < exp_lookup[dE + MAX_DE]) {
        spins[x * L + y] *= -1;
    }
}


// Function to calculate total magnetization
double magnetization(const std::vector<int>& spins) {
    double m = 0;
    for (int i = 0; i < N; ++i) {
        m += spins[i];
    }
    return m / N;
}


// Function to update spins of one color
void update_spins(std::vector<int> &spins, int color, int num_threads, DispatchQueue* dispatch_queue) {
    int spins_per_thread = N / num_threads; 

    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        dispatch_queue->dispatch([&,thread_id] (int) {
            Xorshiro& local_rng = rng;  // Get TLS reference once at the start of thread
            for (int index = thread_id * spins_per_thread; index < (thread_id + 1) * spins_per_thread; ++index) {
                  // Calculate the global index of the spin
                int x = index / L;              // Calculate x coordinate
                int y = index % L;              // Calculate y coordinate

                if ((x + y) % 2 == color) {     // Update only for the specified color
                    monte_carlo_step(spins, x, y, index, local_rng);
                }   
            }
        });
    }
    dispatch_queue->finishTasks();
}

// Function to perform simulation with profiling for initialization and updates
void run_simulation(int num_threads, std::ofstream &prof_file) {
    std::vector<int> spins(N);
    DispatchQueue* dispatch_queue = new  DispatchQueue((size_t) num_threads); // Create a dispatch_queue with num_threads threads

    initialize_exp_lookup(T);
    // Profile initialization
  auto init_start_time = std::chrono::high_resolution_clock::now();  // Start profiling initialization
    initialize_spins(spins);
    auto init_end_time = std::chrono::high_resolution_clock::now();  // End profiling initialization
    std::chrono::duration<double> init_duration = init_end_time - init_start_time;
    // Print initial magnetization
    prof_file << "Step 0 Magnetization: " << magnetization(spins) << "\n";
    
    auto update_start_time = std::chrono::high_resolution_clock::now();
    
    for (int step = 1; step <= steps; ++step) {
        // Update black and white sites separately

        update_spins(spins, 0, num_threads, dispatch_queue);  // Update black sites
        update_spins(spins, 1, num_threads, dispatch_queue);  // Update white sites
        
        if (step % 1 == 0) {
            prof_file << "Step " << step << " Magnetization: " 
                     << magnetization(spins) << "\n";
        }
    }
    auto update_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> update_duration = update_end_time - update_start_time;

    // Multiply by NUM_PARALLEL_SIMS since we're doing 64 simulations at once
    long long total_spin_flips = (long long)steps * N;


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
}

int main() {
    std::cout << "Running Ising model simulation...\n";
    
    // Open CSV file for profiling data
    std::ofstream prof_file("profiling_data_threads.csv");

    for (int num_threads = NUM_THREADS; num_threads <= NUM_THREADS; ++num_threads) {
        run_simulation(num_threads, prof_file);
    }
    
    prof_file.close();
    return 0;
}
