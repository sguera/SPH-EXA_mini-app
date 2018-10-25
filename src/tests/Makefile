CXXFLAGS = -Wall -Wextra -O3 -march=native -msse -msse2 -msse3 -mfpmath=sse -std=c++14
CXX=g++

all: evrard sqpatch tiny-test

evrardBroadTree: ../tree/BroadTree.hpp evrard.cpp
	$(CXX) $(CXXFLAGS) evrard.cpp -D TREEINTERFACE=BroadTree -g -fopenmp -o $@

evrardKdTree: ../tree/KdTree.hpp evrard.cpp
	$(CXX) $(CXXFLAGS) evrard.cpp -D TREEINTERFACE=KdTree -g -fopenmp -o $@

evrardNNFTree: ../tree/NNFTree.hpp evrard.cpp
	$(CXX) $(CXXFLAGS) evrard.cpp -D TREEINTERFACE=NNFTree -g -fopenmp -o $@

sqpatchBroadTree: ../tree/BroadTree.hpp sqpatch.cpp
	$(CXX) $(CXXFLAGS) sqpatch.cpp -D TREEINTERFACE=BroadTree -g -fopenmp -o $@

sqpatchKdTree: ../tree/KdTree.hpp sqpatch.cpp
	$(CXX) $(CXXFLAGS) sqpatch.cpp -D TREEINTERFACE=KdTree -g -fopenmp -o $@

sqpatchNNFTree: ../tree/NNFTree.hpp sqpatch.cpp
	$(CXX) $(CXXFLAGS) sqpatch.cpp -D TREEINTERFACE=NNFTree -g -fopenmp -o $@

tiny-testBroadTree: ../tree/BroadTree.hpp tiny-test.cpp
	$(CXX) $(CXXFLAGS) tiny-test.cpp -D TREEINTERFACE=BroadTree -g -fopenmp -o $@

tiny-testKdTree: ../tree/KdTree.hpp tiny-test.cpp
	$(CXX) $(CXXFLAGS) tiny-test.cpp -D TREEINTERFACE=KdTree -g -fopenmp -o $@

tiny-testNNFTree: ../tree/NNFTree.hpp tiny-test.cpp
	$(CXX) $(CXXFLAGS) tiny-test.cpp -D TREEINTERFACE=NNFTree -g -fopenmp -o $@

.PHONY: evrard sqpatch tiny-test

evrard: evrardBroadTree evrardKdTree evrardNNFTree

sqpatch: sqpatchBroadTree sqpatchKdTree sqpatchNNFTree

tiny-test: tiny-testBroadTree tiny-testKdTree sqpatchNNFTree

clean:
	rm -f *.o evrardBroadTree evrardKdTree evrardNNFTree sqpatchBroadTree sqpatchKdTree sqpatchNNFTree tiny-testBroadTree tiny-testKdTree tiny-testNNFTree
