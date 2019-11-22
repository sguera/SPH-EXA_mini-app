CXX ?= g++ # This is the main compiler
CC ?= gcc
MPICXX ?= mpic++
ENV ?= gnu
NVCC ?= $(CUDA_PATH)/bin/nvcc

CUDA_PATH = /usr/local/cuda

# CXX := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
BINDIR := bin
THIS_FILE := $(lastword $(MAKEFILE_LIST))

#SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
HPP := $(wildcard src/include/*.hpp)
HPP += $(wildcard src/include/tree/*.hpp)

CUDA_OBJS := $(BUILDDIR)/cudaDensity.o $(BUILDDIR)/cudaIAD.o $(BUILDDIR)/cudaMomentumAndEnergyIAD.o $(BUILDDIR)/cudaLookupTables.o

RELEASE := -DNDEBUG
DEBUG := -D__DEBUG -D_GLIBCXX_DEBUG

INC += -Isrc -Isrc/include
CXXFLAGS += $(RELEASE)
NVCCARCH := sm_60
NVCCFLAGS := -std=c++14 --expt-relaxed-constexpr -rdc=true -arch=$(NVCCARCH) --default-stream per-thread -Xcompiler -fopenmp -lgomp
NVCCLDFLAGS := -arch=$(NVCCARCH) -rdc=true

ifeq ($(ENV),gnu)
	CXXFLAGS += -std=c++14 -O2 -Wall -Wextra -fopenmp -fopenacc -march=native -mtune=native -g
endif

ifeq ($(ENV),pgi)
	CXXFLAGS += -O2 -std=c++14 -mp -dynamic -acc -ta=tesla,cc60 -mp=nonuma -Mcuda #-g -Minfo=accel # prints generated accel functions
endif

ifeq ($(ENV),cray)
	CXXFLAGS += -O2 -hstd=c++14 -homp -hacc -dynamic
endif

ifeq ($(ENV),intel)
	CXXFLAGS += -O2 -std=c++14 -qopenmp -dynamic -g
endif

ifeq ($(ENV),clang)
	CXXFLAGS += -O2 -std=c++14 -g -fopenmp=libomp
endif

all: $(TESTCASE)

omp: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CXX) $(CXXFLAGS) $(INC) -DUSE_TASKS src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

mpi+omp: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

omp+cuda: $(BUILDDIR)/cuda_no_mpi.o $(CUDA_OBJS)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(NVCC) $(NVCCLDFLAGS) -DUSE_CUDA -DUSE_TASKS -dlink -o cudalinked.o $(CUDA_OBJS) -lcudadevrt -lcudart -lgomp
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@.app cudalinked.o $+ -L$(CUDA_PATH)/lib64 -lcudart -lcudadevrt -lgomp
#	$(CXX) -o $(BINDIR)/$@.app $+ -L$(CUDA_PATH)/lib64 -lcudart -fopenmp

omp+target: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CXX) $(CXXFLAGS) $(INC) -DUSE_OMP_TARGET src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

mpi+omp+target: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_OMP_TARGET src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

mpi+omp+acc: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_STD_MATH_IN_KERNELS -DUSE_ACC src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

mpi+omp+cuda: $(BUILDDIR)/cuda_mpi.o $(CUDA_OBJS)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(NVCC) $(NVCCLDFLAGS) -dlink -o cudalinked.o $(CUDA_OBJS) -lcudadevrt -lcudart
	$(MPICXX) $(CXXFLAGS) -o $(BINDIR)/$@.app cudalinked.o $+ -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart
#	$(MPICXX) -o $(BINDIR)/$@.app $+ -L$(CUDA_PATH)/lib64 -lcudart -fopenmp

$(BUILDDIR)/cuda_mpi.o: src/sqpatch.cpp
	@mkdir -p $(BUILDDIR)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_CUDA -o $@ -c $<

$(BUILDDIR)/cuda_no_mpi.o: src/sqpatch.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INC) -DUSE_CUDA -DUSE_TASKS -o $@ -c $<

$(BUILDDIR)/%.o: src/include/sph/cuda/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(INC) -c -o $@ $<
#	$(NVCC) $(NVCCFLAGS) $(INC) -DUSE_STD_MATH_IN_KERNELS -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -c -o $@ $<

run_test:
	@$(MAKE) -f $(THIS_FILE) omp
	@$(MAKE) -f $(THIS_FILE) omp+cuda
	@$(MAKE) -f $(THIS_FILE) mpi+omp
	@$(MAKE) -f $(THIS_FILE) mpi+omp+cuda
	cd test/ && ./test_correctness.sh;

clean:
	$(info Cleaning...)
	$(RM) -rf $(BUILDDIR) $(BINDIR)

.PHONY: clean
