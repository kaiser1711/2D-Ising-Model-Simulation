CXX = clang++
METAL_CPP_PATH = $(HOME)/metal-cpp

# Common flags
BASE_CXXFLAGS = -O3 -ffast-math -march=native

# CPU-specific flags
CPU_CXXFLAGS = $(BASE_CXXFLAGS) -std=c++20

# Metal-specific flags and libraries
METAL_CXXFLAGS = -std=c++17 \
 -I$(METAL_CPP_PATH) \
 -I$(METAL_CPP_PATH)/Foundation \
 -I$(METAL_CPP_PATH)/Metal \
 -I$(METAL_CPP_PATH)/QuartzCore
METAL_LDFLAGS = -framework Metal -framework Foundation -framework QuartzCore

# CPU source files
CPU_SOURCES = ising_checkerboard_64_sims.cc \
              ising_basic.cc \
              ising_Xorshiro.cc \
              ising_checkerboard.cc \
              ising_exp_lookup.cc \
              ising_threads.cc

# Generate CPU target names
CPU_TARGETS = $(CPU_SOURCES:.cc=)

# Metal targets
METAL_TARGETS = ising_metal ising_metal_64 default.metallib default_64.metallib

.PHONY: all clean cpu metal

all: cpu metal

cpu: $(CPU_TARGETS)

metal: $(METAL_TARGETS)

# Pattern rule for CPU executables
%: %.cc
	$(CXX) $(CPU_CXXFLAGS) $< -o $@

# Metal executable targets
ising_metal: main.cc
	$(CXX) $(METAL_CXXFLAGS) $< -o $@ $(METAL_LDFLAGS)

ising_metal_64: main_64.cc
	$(CXX) $(METAL_CXXFLAGS) $< -o $@ $(METAL_LDFLAGS)

# Metal shader compilation
default.metallib: ising.metal
	xcrun -sdk macosx metal -c $< -o ising.air
	xcrun -sdk macosx metallib ising.air -o $@

default_64.metallib: ising_64.metal
	xcrun -sdk macosx metal -c $< -o ising_64.air
	xcrun -sdk macosx metallib ising_64.air -o $@

clean:
	rm -f $(CPU_TARGETS) $(METAL_TARGETS) ising.air ising_64.air