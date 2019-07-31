CUDA_PATH ?= /usr/local/cuda

CXX ?= g++ # This is the main compiler
CC ?= gcc
MPICXX ?= mpic++
ENV ?= gnu
NVCC ?= $(CUDA_PATH)/bin/nvcc

# CXX := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
BINDIR := bin
THIS_FILE := $(lastword $(MAKEFILE_LIST))

#SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
HPP := $(wildcard src/include/*.hpp)
HPP += $(wildcard src/include/tree/*.hpp)

CUDA_FILES += $(wildcard src/include/sph/cuda/*.cu)
CUDA_FILES += $(wildcard src/include/sph/cuda/*.cuh)
CUDA_FILES += src/include/sph/kernels.hpp

RELEASE := -DNDEBUG
DEBUG := -D__DEBUG -D_GLIBCXX_DEBUG

INC += -Isrc -Isrc/include
CXXFLAGS += $(RELEASE)
NVCCFLAGS := -std=c++11 --expt-relaxed-constexpr -arch=sm_60

ifeq ($(ENV),gnu)
	CXXFLAGS += -std=c++14 -O2 -Wall -Wextra -fopenmp -fopenacc -march=native -mtune=native
endif

ifeq ($(ENV),pgi)
	CXXFLAGS += -O2 -std=c++14 -mp -dynamic -ta=tesla,cc60 -mp=nonuma
endif

ifeq ($(ENV),cray)
	CXXFLAGS += -O2 -hstd=c++14 -homp -hacc -dynamic
endif

ifeq ($(ENV),intel)
	CXXFLAGS += -O2 -std=c++14 -qopenmp -dynamic
endif

ifeq ($(CXX),clang++)
	COMPILER_VERSION = $(EBVERSIONCLANGPLUSLLVM)
	CXXFLAGS = -std=c++14 -g -fopenmp=libomp
	PE_ENV = CLANG
	LDFLAGS += $(CXXFLAGS) -dynamic
endif

all: $(TESTCASE)

omp: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CXX) $(CXXFLAGS) $(INC) src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

mpi+omp: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

omp+cuda: $(BUILDDIR)/cuda_no_mpi.o $(BUILDDIR)/cuda.o
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CXX) -o $(BINDIR)/$@.app $+ -L$(CUDA_PATH)/lib64 -lcudart -fopenmp

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
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_ACC src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

mpi+omp+cuda: $(BUILDDIR)/cuda_mpi.o $(BUILDDIR)/cuda.o
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) -o $(BINDIR)/$@.app $+ -L$(CUDA_PATH)/lib64 -lcudart -fopenmp

$(BUILDDIR)/cuda_mpi.o: src/sqpatch.cpp
	@mkdir -p $(BUILDDIR)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_CUDA -o $@ -c $<

$(BUILDDIR)/cuda_no_mpi.o: src/sqpatch.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INC) -DUSE_CUDA -o $@ -c $<

$(BUILDDIR)/cuda.o: $(CUDA_FILES)
	@echo $(CUDA_FILES)
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $(INC) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -DUSE_STD_MATH_IN_KERNELS -c -o $@ $<

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
