CXXFLAGS = -Wall -Wextra -O3 -march=native -msse -msse2 -msse3 -mfpmath=sse -std=c++14
CXX=g++

all: evrard sqpatch tiny-test

evrardBroadTree:
	$(CXX) $(CXXFLAGS) evrard.cpp -D TREEINTERFACE=BroadTree -g -fopenmp -o $@

evrardKdTree:
	$(CXX) $(CXXFLAGS) evrard.cpp -D TREEINTERFACE=KdTree -g -fopenmp -o $@

sqpatchBroadTree:
	$(CXX) $(CXXFLAGS) sqpatch.cpp -D TREEINTERFACE=BroadTree -g -fopenmp -o $@

sqpatchKdTree:
	$(CXX) $(CXXFLAGS) sqpatch.cpp -D TREEINTERFACE=KdTree -g -fopenmp -o $@

tiny-testBroadTree:
	$(CXX) $(CXXFLAGS) tiny-test.cpp -D TREEINTERFACE=BroadTree -g -fopenmp -o $@

tiny-testKdTree:
	$(CXX) $(CXXFLAGS) tiny-test.cpp -D TREEINTERFACE=KdTree -g -fopenmp -o $@

.PHONY: evrard sqpatch tiny-test

evrard: evrardBroadTree evrardKdTree

sqpatch: sqpatchBroadTree sqpatchKdTree

tiny-test: tiny-testBroadTree tiny-testKdTree

clean:
	rm -f *.o evrardBroadTree evrardKdTree sqpatchBroadTree sqpatchKdTree tiny-testBroadTree tiny-testKdTree
