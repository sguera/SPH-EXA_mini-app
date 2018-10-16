CXXFLAGS = -Wall -Wextra -O3 -march=native -msse -msse2 -msse3 -mfpmath=sse
CXX=g++

all:
	$(CXX) $(CXXFLAGS) Tree.cpp main.cpp -fopenmp

clean:
	rm -f *.o