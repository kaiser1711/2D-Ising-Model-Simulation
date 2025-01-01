#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>
#include <chrono>
#include <random>

const bool PRINT_MAGNETIZATION = 0;

struct SimParams {
    uint32_t L;  
    int J;
    float T;
    uint32_t seed;
};

class IsingSimulation {
private:
    // Metal objects
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* initPipelineState;
    MTL::ComputePipelineState* updatePipelineState;
    MTL::ComputePipelineState* magPipelineState;
    
    // Simulation parameters
    const uint32_t L = 20000;  // Lattice size
    const int N;
    const int steps = 10;
    const int J = 1;
    const float T = 2.0f;
    
    // Metal buffers
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
    IsingSimulation() : N(L * L) {
        // Get default Metal device
        device = MTL::CreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("Failed to create Metal device");
        
        // Create command queue
        commandQueue = device->newCommandQueue();
        if (!commandQueue) throw std::runtime_error("Failed to create command queue");
        
        // Load metal library from default.metallib
        NS::Error* error = nullptr;
        auto library = device->newDefaultLibrary();
        if (!library) throw std::runtime_error("Failed to load Metal library");
        
        createPipelineStates(library);
        
        // Create buffers
        spinBuffer = device->newBuffer(N * sizeof(int), MTL::ResourceStorageModeShared);
        magBuffer = device->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
        if (!spinBuffer || !magBuffer) throw std::runtime_error("Failed to create buffers");
    }
    
    ~IsingSimulation() {
        spinBuffer->release();
        magBuffer->release();
        initPipelineState->release();
        updatePipelineState->release();
        magPipelineState->release();
        commandQueue->release();
        device->release();
    }
    
    long long getTotalSpinFlips() {return (long long) N*steps;}
   

    void run() {
        // Create simulation parameters
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

        int totalMag;
        float magnetization;
        int zeroMag;
        MTL::Size magGridSize = MTL::Size(N, 1, 1);
        MTL::Size magThreadgroupSize = MTL::Size(256, 1, 1);

        if(PRINT_MAGNETIZATION) 
        {
            // Print initial magnetization
            zeroMag = 0;
            memcpy(magBuffer->contents(), &zeroMag, sizeof(int));
            
            commandBuffer = commandQueue->commandBuffer();
            encoder = commandBuffer->computeCommandEncoder();
            encoder->setComputePipelineState(magPipelineState);
            encoder->setBuffer(spinBuffer, 0, 0);
            encoder->setBuffer(magBuffer, 0, 1);
        
            encoder->dispatchThreads(magGridSize, magThreadgroupSize);
            
            encoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
            
            totalMag = *static_cast<int*>(magBuffer->contents());
            magnetization = static_cast<float>(totalMag) / N;
            std::cout << "Initial Magnetization: " << magnetization << std::endl;
        }

        // Main simulation loop
        for (int step = 1; step <= steps; ++step) {
            // Update black cells
            commandBuffer = commandQueue->commandBuffer();
            encoder = commandBuffer->computeCommandEncoder();
            encoder->setComputePipelineState(updatePipelineState);
            
            uint32_t color = 0;
            encoder->setBytes(&color, sizeof(color), 2);
            encoder->setBuffer(spinBuffer, 0, 0);
            encoder->setBuffer(paramsBuffer, 0, 1);
            encoder->dispatchThreads(gridSize, threadgroupSize);
            
            encoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
            
            // Update white cells
            commandBuffer = commandQueue->commandBuffer();
            encoder = commandBuffer->computeCommandEncoder();
            encoder->setComputePipelineState(updatePipelineState);
            
            color = 1;
            encoder->setBytes(&color, sizeof(color), 2);
            encoder->setBuffer(spinBuffer, 0, 0);
            encoder->setBuffer(paramsBuffer, 0, 1);
            encoder->dispatchThreads(gridSize, threadgroupSize);
            
            encoder->endEncoding();
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();

            if(step % 1 == 0 && PRINT_MAGNETIZATION)
            {
                // Calculate magnetization
                memcpy(magBuffer->contents(), &zeroMag, sizeof(int));
                
                commandBuffer = commandQueue->commandBuffer();
                encoder = commandBuffer->computeCommandEncoder();
                encoder->setComputePipelineState(magPipelineState);
                encoder->setBuffer(spinBuffer, 0, 0);
                encoder->setBuffer(magBuffer, 0, 1);
                encoder->dispatchThreads(magGridSize, magThreadgroupSize);
                
                encoder->endEncoding();
                commandBuffer->commit();
                commandBuffer->waitUntilCompleted();
                
                totalMag = *static_cast<int*>(magBuffer->contents());
                magnetization = static_cast<float>(totalMag) / N;
                std::cout << "Step " << step << " Magnetization: " << magnetization << std::endl;
            }
        }
        
        paramsBuffer->release();
    }
};

int main() {
    try {
        std::cout << "Starting Ising model simulation on Metal..." << std::endl;
        IsingSimulation simulation;
        
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