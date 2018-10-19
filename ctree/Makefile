CXXFLAGS = -Wall -Wextra -O3 -march=native -msse -msse2 -msse3 -mfpmath=sse -std=c++14
CXX=g++

all:
	$(CXX) $(CXXFLAGS) Tree.cpp main.cpp -g -fopenmp -o evrad
	$(CXX) $(CXXFLAGS) Tree.cpp tiny_test.cpp -g -fopenmp -o tiny-test

clean:
	rm -f *.o
