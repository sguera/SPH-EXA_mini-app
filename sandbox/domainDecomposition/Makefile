ENV = gnu

ifeq ($(ENV),gnu)
	CXX = mpic++
	COMPILER = gnu
	CXXFLAGS += -g -O3 -Wall -Wextra -std=c++14 -fopenmp
endif

ifeq ($(ENV),intel)
	CXX = mpiicc
	COMPILER = intel
	CXXFLAGS += -g -O3 -Wall -Wextra -rdynamic -std=c++14 -qopenmp
endif

all: main

main:
	$(CXX) $(CXXFLAGS) main.cpp -o $@.$(COMPILER)

N=4
run: main
	mpirun --hostfile hosts -n $(N) ./a.out
