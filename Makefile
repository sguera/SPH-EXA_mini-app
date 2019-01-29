CXX := g++ # This is the main compiler
MPICXX := mpic++
ENV := gnu

# CXX := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
BINDIR := bin
 
#SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
HPP := $(wildcard src/include/*.hpp)
HPP += $(wildcard src/include/tree/*.hpp)


CXXFLAGS += -I src/include
DEBUG := -D__DEBUG -D_GLIBCXX_DEBUG

ifeq ($(ENV),gnu)
	CXXFLAGS += -std=c++14 -O2 -s -g -Wall -Wextra -fopenmp -march=native -mtune=native
endif

ifeq ($(ENV),pgi)
	CXXFLAGS += -O2 -std=c++14 -g -mp -dynamic
endif

ifeq ($(ENV),cray)
	CXXFLAGS += -O2 -hstd=c++14 -g -homp -dynamic
endif

ifeq ($(ENV),intel)
	CXXFLAGS += -O2 -std=c++14 -g -qopenmp -dynamic
endif

ifeq ($(CXX),clang++)
	COMPILER_VERSION = $(EBVERSIONCLANGPLUSLLVM)
	CXXFLAGS = -std=c++14 -g -fopenmp=libomp
	PE_ENV = CLANG
	LDFLAGS += $(CXXFLAGS) -dynamic
endif

all: $(TESTCASE)

evrard: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CXX) $(CXXFLAGS) $(INC) src/evrard.cpp -o $(BINDIR)/$@.app $(LIB)

mpievrard: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI src/evrard.cpp -o $(BINDIR)/$@.app $(LIB)

sqpatch: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CXX) $(CXXFLAGS) $(INC) src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

mpisqpatch: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICXX) $(CXXFLAGS) $(INC) -DUSE_MPI src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

run: evrard

clean:
	$(info Cleaning...) 
	$(RM) -rf $(BUILDDIR) $(BINDIR)

.PHONY: clean
