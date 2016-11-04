# Makefile for C/C++ programs.

# Author: John M. Weiss, Ph.D.

# Usage:  make target1 target2 ...

#-----------------------------------------------------------------------

# GNU C/C++ compiler and linker:
CC = gcc
CXX = g++
LINK = nvcc

# Turn on optimization and warnings (add -g for debugging with gdb):
# CPPFLAGS = 		# preprocessor flags
CFLAGS = -O -Wall -std=c++11 -fopenmp
CXXFLAGS = -O -Wall -std=c++11 -fopenmp
LIBS = -fopenmp

#-----------------------------------------------------------------------

# MAKE allows the use of "wildcards", to make writing compilation instructions
# a bit easier. GNU make uses $@ for the target and $^ for the dependencies.

all:    primes

# specific targets
gpgpuPrimes.o:
	nvcc -c gpgpuPrimes.cu -std=c++11

primes:	gpgpuPrimes.o primes.o primeCheck.o 
	$(LINK) -o $@ $^ -Xcompiler "$(LIBS)"



# typical target entry, builds "myprog" from file1.cpp, file2.cpp, file3.cpp
myprog:	file1.o file2.o file3.o
	$(LINK) -o $@ $^  -Xcompiler "$(LIBS)"

# generic C and C++ targets for OpenGL programs consisting of only one file
# type "make filename" (no extension) to build
.c:
	$(CC) -o $@ $@.c $(CFLAGS) $(LIBS)

.cpp:
	$(CXX) -o $@ $@.cpp $(CXXFLAGS) $(LIBS)

# utility targets
clean:
	rm -f *.o *~ core
