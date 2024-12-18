CXX = clang++
CXXFLAGS = -O3 -ffast-math -march=native -std=c++20

# List all your source files here
SOURCES = ising_checkerboard_64_sims.cc \
		  ising_basic.cc \
		  ising_Xorshiro.cc \
		  ising_checkerboard.cc \
		  ising_exp_lookup.cc \
		  ising_threads.cc \
          # Add more source files as needed

# Generate target names by removing .cc extension
TARGETS = $(SOURCES:.cc=)

.PHONY: all clean

all: $(TARGETS)

# Pattern rule for building executables
%: %.cc
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
