CXX = g++
CXXFLAGS = -O3 -ffast-math -march=native -mprefer-vector-width=256 -flto -fomit-frame-pointer -falign-loops=32 -std=c++17 -fopenmp

all: nbody

nbody: nbody.cpp
	# Compile with profile generation
	$(CXX) $(CXXFLAGS) -fprofile-generate -o nbody nbody.cpp
	# Run the program with the actual input data to collect profile info
	OMP_NUM_THREADS=4 ./nbody ./generation_0_100k.csv dummy_pgo.csv 1 0.01
	# Clean up dummy output
	rm -f dummy_pgo.csv
	# Recompile using the collected profiles
	$(CXX) $(CXXFLAGS) -fprofile-use -fprofile-correction -o nbody nbody.cpp
	# Clean up profile files
	rm -f *.gcda *.gcno

clean:
	rm -f nbody *.gcda *.gcno
