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

CUDA_OBJS := $(BUILDDIR)/gather.o $(BUILDDIR)/findneighbors.o $(BUILDDIR)/cudaDensity.o $(BUILDDIR)/cudaIAD.o $(BUILDDIR)/cudaMomentumAndEnergyIAD.o

RELEASE := -DNDEBUG
DEBUG := -D__DEBUG -D_GLIBCXX_DEBUG

# cuda architecture targets
SMS ?= 35 60 70 75
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

INC += -Isrc -Iinclude -Idomain/include -I$(CUDA_PATH)/include
CXXFLAGS += $(RELEASE)
NVCCFLAGS := -std=c++14 --expt-relaxed-constexpr -rdc=true $(GENCODE_FLAGS) -Wno-deprecated-gpu-targets -g
NVCCLDFLAGS := $(GENCODE_FLAGS) -rdc=true

ifeq ($(ENV),gnu)
	CXXFLAGS += -std=c++17 -O2 -Wall -Wextra -fopenmp -fopenacc -march=native -mtune=native -g
endif

ifeq ($(ENV),pgi)
	CXXFLAGS += -O2 -std=c++17 -mp -dynamic -acc -ta=tesla,cc60 -mp=nonuma -Mcuda -g #-g -Minfo=accel # prints generated accel functions
endif

ifeq ($(ENV),cray)
	CXXFLAGS += -O2 -hstd=c++17 -homp -hacc -dynamic -g
endif

ifeq ($(ENV),intel)
	CXXFLAGS += -O2 -std=c++17 -qopenmp -dynamic -g
endif

ifeq ($(ENV),clang)
	CXXFLAGS += -O2 -march=native -std=c++17 -fopenmp -g
endif

TESTCASE ?= sedov

ifeq ($(TESTCASE),evrard)
	TESTCASE_FLAGS = -DGRAVITY
endif

#omp: $(HPP)
#	@mkdir -p $(BINDIR)
#	$(info Linking the executable:)
#	$(CXX) $(CXXFLAGS) $(INC) $(TESTCASE_FLAGS) src/$(TESTCASE)/$(TESTCASE).cpp -o $(BINDIR)/$@.app $(LIB)

mpi+omp: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI $(TESTCASE_FLAGS) src/$(TESTCASE)/$(TESTCASE).cpp -o $(BINDIR)/$@.app $(LIB)

#omp+cuda: $(BUILDDIR)/cuda_no_mpi.o $(CUDA_OBJS)
#	@mkdir -p $(BINDIR)
#	$(info Linking the executable:)
#	$(NVCC) $(NVCCLDFLAGS) -DUSE_CUDA $(TESTCASE_FLAGS) -dlink -o cudalinked.o $(CUDA_OBJS) -lcudadevrt -lcudart
#	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@.app cudalinked.o $+ -L$(CUDA_PATH)/lib64 -lcudart -lcudadevrt
##	$(CXX) -o $(BINDIR)/$@.app $+ -L$(CUDA_PATH)/lib64 -lcudart -fopenmp

#omp+target: $(HPP)
#	@mkdir -p $(BINDIR)
#	$(info Linking the executable:)
#	$(CXX) $(CXXFLAGS) $(INC) -DUSE_OMP_TARGET $(TESTCASE_FLAGS) src/$(TESTCASE)/$(TESTCASE).cpp -o $(BINDIR)/$@.app $(LIB)

mpi+omp+target: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_OMP_TARGET $(TESTCASE_FLAGS) src/$(TESTCASE)/$(TESTCASE).cpp -o $(BINDIR)/$@.app $(LIB)

mpi+omp+acc: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_STD_MATH_IN_KERNELS $(TESTCASE_FLAGS) -DUSE_ACC src/$(TESTCASE)/$(TESTCASE).cpp -o $(BINDIR)/$@.app $(LIB)

mpi+omp+cuda: $(BUILDDIR)/cuda_mpi.o $(CUDA_OBJS)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(NVCC) $(NVCCLDFLAGS) -dlink -o cudalinked.o $(CUDA_OBJS) -lcudadevrt -lcudart
	$(MPICXX) $(CXXFLAGS) -o $(BINDIR)/$@.app cudalinked.o $+ -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart

#all: omp mpi+omp omp+cuda mpi+omp+cuda omp+target mpi+omp+target mpi+omp+acc
all: mpi+omp mpi+omp+cuda mpi+omp+target mpi+omp+acc

$(BUILDDIR)/cuda_mpi.o: src/$(TESTCASE)/$(TESTCASE).cpp
	@mkdir -p $(BUILDDIR)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI -DUSE_CUDA $(TESTCASE_FLAGS) -o $@ -c $<

$(BUILDDIR)/cuda_no_mpi.o: src/$(TESTCASE)/$(TESTCASE).cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INC) -DUSE_CUDA $(TESTCASE_FLAGS) -o $@ -c $<

$(BUILDDIR)/%.o: include/sph/cuda/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(TESTCASE_FLAGS) $(INC) -c -o $@ $<
#	$(NVCC) $(NVCCFLAGS) $(INC) -DUSE_STD_MATH_IN_KERNELS -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -c -o $@ $<

$(BUILDDIR)/%.o: domain/include/cstone/cuda/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(TESTCASE_FLAGS) $(INC) -c -o $@ $<

run_test:
#	@$(MAKE) -f $(THIS_FILE) omp
#	@$(MAKE) -f $(THIS_FILE) omp+cuda
	@$(MAKE) -f $(THIS_FILE) mpi+omp
	@$(MAKE) -f $(THIS_FILE) mpi+omp+cuda
	cd test/ && ./test_correctness.sh;

clean:
	$(info Cleaning...)
	$(RM) -rf $(BUILDDIR) $(BINDIR)

.PHONY: all clean
