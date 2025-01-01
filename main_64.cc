#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <chrono>
#include <random>

const bool PRINT_MAGNETIZATION = 0;
// Ensure struct alignment matches between CPU and GPU
struct alignas(64) SpinBlock {
    uint64_t spins;  // 64 parallel simulations
    uint64_t padding[7];  // Pad to cache line size
};

struct SimParams {
    uint32_t L;
    int J;
    float T;
    uint32_t seed;
};

class ParallelIsingSimulation {
private:
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* initPipelineState;
    MTL::ComputePipelineState* updatePipelineState;
    MTL::ComputePipelineState* magPipelineState;
    
    const uint32_t L = 20000;  // Lattice size
    const int N;  // Total number of sites
    const int steps = 10;
    const int J = 1;
    const float T = 2.0f;
    const int NUM_PARALLEL_SIMS = 64;
    
    MTL::Buffer* spinBuffer;
    MTL::Buffer* magBuffer;
    
    void createPipelineStates(MTL::Library* library) {
        NS::Error* error = nullptr;
        
        auto initFunction = library->newFunction(NS::String::string("initialize_spins", NS::UTF8StringEncoding));
        initPipelineState = device->newComputePipelineState(initFunction, &error);
        if (!initPipelineState) throw std::runtime_error("Failed to create init pipeline state");
        
        auto updateFunction = library->newFunction(NS::String::string("update_spins", NS::UTF8StringEncoding));
        updatePipelineState = device->newComputePipelineState(updateFunction, &error);
        if (!updatePipelineState) throw std::runtime_error("Failed to create update pipeline state");
        
        auto magFunction = library->newFunction(NS::String::string("calculate_magnetization", NS::UTF8StringEncoding));
        magPipelineState = device->newComputePipelineState(magFunction, &error);
        if (!magPipelineState) throw std::runtime_error("Failed to create magnetization pipeline state");
    }

public:
    ParallelIsingSimulation() : N(L * L) {
        device = MTL::CreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("Failed to create Metal device");
        
        commandQueue = device->newCommandQueue();
        if (!commandQueue) throw std::runtime_error("Failed to create command queue");
        
        NS::Error* error = nullptr;
        NS::String* path = NS::String::string("default_64.metallib", NS::ASCIIStringEncoding);
        auto library = device->newLibrary(path, &error);
        if (!library) {
            if (error) {
                std::string err = error->localizedDescription()->utf8String();
                throw std::runtime_error("Failed to load Metal library: " + err);
            }
            throw std::runtime_error("Failed to load Metal library");
        }
        
        createPipelineStates(library);
        
        // Allocate memory for SpinBlocks (64 bits per site)
        spinBuffer = device->newBuffer(N * sizeof(SpinBlock), MTL::ResourceStorageModeShared);
        // Allocate separate magnetization counters for each simulation
        magBuffer = device->newBuffer(NUM_PARALLEL_SIMS * sizeof(int), MTL::ResourceStorageModeShared);
        
        if (!spinBuffer || !magBuffer) throw std::runtime_error("Failed to create buffers");
    }
    
    ~ParallelIsingSimulation() {
        spinBuffer->release();
        magBuffer->release();
        initPipelineState->release();
        updatePipelineState->release();
        magPipelineState->release();
        commandQueue->release();
        device->release();
    }
    
    long long getTotalSpinFlips() {
        return (long long)N * steps * NUM_PARALLEL_SIMS;
    }

    void run() {
        std::random_device rd;
        SimParams params{L, J, T, static_cast<uint32_t>(rd())};
        auto paramsBuffer = device->newBuffer(&params, sizeof(SimParams), MTL::ResourceStorageModeShared);
        
        // Initialize spins
        auto commandBuffer = commandQueue->commandBuffer();
        auto encoder = commandBuffer->computeCommandEncoder();
        
        encoder->setComputePipelineState(initPipelineState);
        encoder->setBuffer(spinBuffer, 0, 0);
        encoder->setBuffer(paramsBuffer, 0, 1);
        
        MTL::Size gridSize = MTL::Size(L, L, 1);
        MTL::Size threadgroupSize = MTL::Size(16, 16, 1);
        encoder->dispatchThreads(gridSize, threadgroupSize);
        
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        std::vector<float> magnetizations;
        if(PRINT_MAGNETIZATION){
            // Print initial magnetization for each simulation
             magnetizations = calculateMagnetization();
            std::cout << "Initial Magnetizations:\n";
            for (int sim = 0; sim < NUM_PARALLEL_SIMS; sim++) {
                if (sim < 5) { // Print first 5 simulations only to avoid clutter
                    std::cout << "Sim " << sim << ": " << magnetizations[sim] << "\n";
                }
            }
        }     
        // Main simulation loop
        for (int step = 1; step <= steps; ++step) {
            // Update black sites
            updateColor(0, paramsBuffer);
            
            // Update white sites
            updateColor(1, paramsBuffer);
            
            if(step % 10 == 0 && PRINT_MAGNETIZATION)
            {
                // Calculate and print magnetization
                magnetizations = calculateMagnetization();
                std::cout << "Step " << step << " Magnetizations:\n";
                for (int sim = 0; sim < NUM_PARALLEL_SIMS; sim++) {
                    if (sim < 5) { // Print first 5 simulations only
                        std::cout << "Sim " << sim << ": " << magnetizations[sim] << "\n";
                    }
                }
            }
        }
        
        paramsBuffer->release();
    }

private:
    void updateColor(uint32_t color, MTL::Buffer* paramsBuffer) {
        auto commandBuffer = commandQueue->commandBuffer();
        auto encoder = commandBuffer->computeCommandEncoder();
        
        encoder->setComputePipelineState(updatePipelineState);
        encoder->setBuffer(spinBuffer, 0, 0);
        encoder->setBuffer(paramsBuffer, 0, 1);
        encoder->setBytes(&color, sizeof(color), 2);
        
        // Compute probabilities
        float prob_pos4 = std::exp(-4.0f * J / T);
        float prob_pos8 = std::exp(-8.0f * J / T);

        // Pass them as arguments to the Metal kernel
        encoder->setBytes(&prob_pos4, sizeof(float), 3);
        encoder->setBytes(&prob_pos8, sizeof(float), 4);
                
        MTL::Size gridSize = MTL::Size(L, L, 1);
        MTL::Size threadgroupSize = MTL::Size(16, 16, 1);
        encoder->dispatchThreads(gridSize, threadgroupSize);
        
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    }
    
    std::vector<float> calculateMagnetization() {
        // Reset magnetization counters
        std::vector<int> zeroMags(NUM_PARALLEL_SIMS, 0);
        memcpy(magBuffer->contents(), zeroMags.data(), NUM_PARALLEL_SIMS * sizeof(int));
        
        auto commandBuffer = commandQueue->commandBuffer();
        auto encoder = commandBuffer->computeCommandEncoder();
        
        encoder->setComputePipelineState(magPipelineState);
        encoder->setBuffer(spinBuffer, 0, 0);
        encoder->setBuffer(magBuffer, 0, 1);
        
        MTL::Size gridSize = MTL::Size(N, 1, 1);
        MTL::Size threadgroupSize = MTL::Size(256, 1, 1);
        encoder->dispatchThreads(gridSize, threadgroupSize);
        
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        // Read back magnetizations for all simulations
        std::vector<float> magnetizations(NUM_PARALLEL_SIMS);
        int* magData = static_cast<int*>(magBuffer->contents());
        for (int i = 0; i < NUM_PARALLEL_SIMS; i++) {
            magnetizations[i] = static_cast<float>(magData[i]) / N;
        }
        
        return magnetizations;
    }
};

int main() {
    try {
        std::cout << "Starting 64-parallel Ising model simulation on Metal..." << std::endl;
        ParallelIsingSimulation simulation;
        
        auto start = std::chrono::high_resolution_clock::now();
        simulation.run();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> duration = end - start;
        std::cout << "Simulation completed in " << duration.count() << " seconds" << std::endl;
        std::cout << duration.count() << "," << simulation.getTotalSpinFlips() / duration.count() << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}