CC = gcc
CFLAGS = -O3 -ffast-math -march=native -Wall -Wextra -std=gnu99 --param ggc-min-expand=0 --param ggc-min-heapsize=1024

all: nbody

nbody: nbody.cpp
	$(CC) -x c $(CFLAGS) nbody.cpp -o nbody -lm

clean:
	rm -f nbody output.csv time_output.txt

