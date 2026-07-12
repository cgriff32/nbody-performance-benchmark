CXX = g++
CXXFLAGS = -O3 -ffast-math -march=native -funroll-loops -flto -fomit-frame-pointer -falign-loops=32 -std=c++17

all: nbody

nbody: nbody.cpp
	$(CXX) $(CXXFLAGS) -o nbody nbody.cpp

clean:
	rm -f nbody
