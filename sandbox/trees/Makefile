CXXFLAGS = -Wall -Wextra -O3 -march=native -msse -msse2 -msse3 -mfpmath=sse -std=c++14
CXX=g++

all: tiny-test treeTest treePerf

tiny-testBroadTree: ../../src/tree/BroadTree.hpp tiny-test.cpp
	$(CXX) $(CXXFLAGS) tiny-test.cpp -D USE_H -D TREEINTERFACE=BroadTree -g -fopenmp -o $@

tiny-testKdTree: ../../src/tree/KdTree.hpp tiny-test.cpp
	$(CXX) $(CXXFLAGS) tiny-test.cpp -D TREEINTERFACE=KdTree -g -fopenmp -o $@

tiny-testOctree: ../../src/tree/Octree.hpp tiny-test.cpp
	$(CXX) $(CXXFLAGS) tiny-test.cpp -D TREEINTERFACE=Octree -g -fopenmp -o $@

tiny-testNNFTree: ../../src/tree/NNFTree.hpp tiny-test.cpp
	$(CXX) $(CXXFLAGS) tiny-test.cpp -D TREEINTERFACE=NNFTree -g -fopenmp -o $@
	
TESTCASE=EVRARD
treeTestBroadTree: treeTest.cpp ../../src/tree/BroadTree.hpp
	$(CXX) $(CXXFLAGS) treeTest.cpp -D $(TESTCASE) -D USE_H -D TREEINTERFACE=BroadTree -g -fopenmp -o $@

treeTestKdTree: treeTest.cpp ../../src/tree/KdTree.hpp 
	$(CXX) $(CXXFLAGS) treeTest.cpp -D $(TESTCASE) -D TREEINTERFACE=KdTree -g -fopenmp -o $@

treeTestOctree: treeTest.cpp ../../src/tree/Octree.hpp 
	$(CXX) $(CXXFLAGS) treeTest.cpp -D $(TESTCASE) -D TREEINTERFACE=Octree -g -fopenmp -o $@

treeTestNNFTree: treeTest.cpp ../../src/tree/NNFTree.hpp 
	$(CXX) $(CXXFLAGS) treeTest.cpp -D $(TESTCASE) -D TREEINTERFACE=NNFTree -g -fopenmp -o $@

REPEAT=100
treePerfBroadTree: treePerf.cpp ../../src/tree/BroadTree.hpp
	$(CXX) $(CXXFLAGS) treePerf.cpp -D REPEAT=$(REPEAT) -D $(TESTCASE) -D USE_H -D TREEINTERFACE=BroadTree -g -fopenmp -o $@

treePerfKdTree: treePerf.cpp ../../src/tree/KdTree.hpp 
	$(CXX) $(CXXFLAGS) treePerf.cpp -D REPEAT=$(REPEAT) -D $(TESTCASE) -D TREEINTERFACE=KdTree -g -fopenmp -o $@

treePerfOctree: treePerf.cpp ../../src/tree/Octree.hpp 
	$(CXX) $(CXXFLAGS) treePerf.cpp -D REPEAT=$(REPEAT) -D $(TESTCASE) -D TREEINTERFACE=Octree -g -fopenmp -o $@

treePerfNNFTree: treePerf.cpp ../../src/tree/NNFTree.hpp 
	$(CXX) $(CXXFLAGS) treePerf.cpp -D REPEAT=$(REPEAT) -D $(TESTCASE) -D TREEINTERFACE=NNFTree -g -fopenmp -o $@

.PHONY: tiny-test treeTest treePerf

tiny-test: tiny-testBroadTree tiny-testKdTree tiny-testNNFTree tiny-testOctree

treeTest: treeTestBroadTree treeTestKdTree treeTestNNFTree treeTestOctree

treePerf: treePerfBroadTree treePerfKdTree treePerfNNFTree treePerfOctree

clean:
	rm -f *.o 
	rm -f tiny-testBroadTree tiny-testKdTree tiny-testNNFTree tiny-testOctree
	rm -f treeTestBroadTree treeTestKdTree treeTestNNFTree treeTestOctree
	rm -f treePerfBroadTree treePerfKdTree treePerfNNFTree treePerfOctree
