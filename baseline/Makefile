CXX = g++
CXXFLAGS = -O0 -Wall -Wextra -std=c++17

all: nbody

nbody: nbody.cpp
	$(CXX) $(CXXFLAGS) nbody.cpp -o nbody

clean:
	rm -f nbody output.csv time_output.txt
