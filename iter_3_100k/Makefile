CXX = g++
CXXFLAGS = -O3 -Ofast -march=native -fopenmp -Wall -Wextra -std=c++17
CC = gcc
CFLAGS = -O3 -Ofast -march=native -fopenmp -shared -fPIC

all: nbody fast_energy.so

nbody: nbody.cpp
	$(CXX) $(CXXFLAGS) nbody.cpp -o nbody

fast_energy.so: fast_energy.c
	$(CC) $(CFLAGS) fast_energy.c -o fast_energy.so

clean:
	rm -f nbody fast_energy.so output.csv time_output.txt

