#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>


const int L = 20000;      // Lattice size
const int N = L * L;     // Number of spins
const int steps = 10;   // Number of Monte Carlo steps
const int J = 1;         // Interaction strength
const double T = 2.0;    // Temperature (in units of J/k_B)


// Pre-compute exponentials for possible energy changes
constexpr int MAX_DE = 8 * J;
std::vector<double> exp_lookup(2 * MAX_DE + 1);

void initialize_exp_lookup(double T) {
    for (int dE = -MAX_DE; dE <= MAX_DE; ++dE) {
        exp_lookup[dE + MAX_DE] = exp(-dE / T);
    }
}

void initialize_spins(std::vector<int>& spins) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < N; ++i) {
        spins[i] = (dis(gen) < 0.25) ? -1 : 1;
    }
}

int periodic(int i, int limit) {
    return (i + limit) % limit;
}

int delta_energy(const std::vector<int>& spins, int x, int y) {
    int spin = spins[x * L + y];
    int neighbors = 
        spins[periodic(x-1, L) * L + y] +
        spins[periodic(x+1, L) * L + y] +
        spins[x * L + periodic(y-1, L)] +
        spins[x * L + periodic(y+1, L)];
    
    return 2 * J * spin * neighbors;
}

double magnetization(const std::vector<int>& spins) {
    double m = 0;
    for (int i = 0; i < N; ++i) {
        m += spins[i];
    }
    return m / N;
}

// Update spins of one color (checkerboard pattern)
void update_spins(std::vector<int>& spins, int color, std::mt19937& gen) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            if ((x + y) % 2 == color) {
                int dE = delta_energy(spins, x, y);
                
                // Use pre-computed exponentials
                if (dE <= 0 || dis(gen) < exp_lookup[dE + MAX_DE]) {
                    spins[x * L + y] *= -1;
                }
            }
        }
    }
}

void run_simulation(std::ofstream &prof_file) {
    std::vector<int> spins(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    auto init_start_time = std::chrono::high_resolution_clock::now();
    // Initialize the system
    initialize_spins(spins);
    initialize_exp_lookup(T);
    auto init_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> init_duration = init_end_time - init_start_time;
    // Print initial magnetization
    prof_file << "Step 0 Magnetization: " << magnetization(spins) << "\n";
    
    auto update_start_time = std::chrono::high_resolution_clock::now();
    
    for (int step = 1; step <= steps; ++step) {
        // Update black and white sites separately
        update_spins(spins, 0, gen);  // Update black sites
        update_spins(spins, 1, gen);  // Update white sites
        
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
    prof_file << 1 << "," 
              << init_duration.count() << "," 
              << update_duration.count() << ","
              << total_spin_flips / update_duration.count() << "\n";
    std::cout << 1 << "," 
              << init_duration.count() << "," 
              << update_duration.count() << ","
              << total_spin_flips / update_duration.count() << "\n";
}

int main() {
    std::cout << "Running Ising model simulation...\n";
    
    std::ofstream prof_file("profiling_data_exp_lookup.csv");
    run_simulation(prof_file);
    
    prof_file.close();
    return 0;
}