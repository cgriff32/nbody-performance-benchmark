CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -march=native -ffast-math -funroll-loops

all: nbody

nbody: nbody.cpp
	$(CXX) $(CXXFLAGS) nbody.cpp -o nbody

clean:
	rm -f nbody output.csv time_output.txt

