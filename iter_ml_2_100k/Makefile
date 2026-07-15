CXX = g++
CXXFLAGS = -O3 -ffast-math -march=native -fopenmp -fPIC -shared

all: libnbody.so
	chmod +x nbody

libnbody.so: nbody_kern.cpp
	$(CXX) $(CXXFLAGS) nbody_kern.cpp -o libnbody.so

clean:
	rm -f output.csv time_output.txt simulation_time.txt libnbody.so
