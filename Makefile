CC := g++ # This is the main compiler
MPICC := mpic++

# CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
BINDIR := bin
 
#SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
HPP := $(wildcard src/include/*.hpp)
HPP += $(wildcard src/include/tree/*.hpp)

USE_MPI=
CFLAGS := -std=c++14 -O2 -s -Wall -Wextra -fopenmp -march=native -mtune=native $(USE_MPI)
DEBUG := -D__DEBUG -D_GLIBCXX_DEBUG
INC := -I src/include
LIB := 

all: $(TESTCASE)

evrard: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CC) $(CFLAGS) $(INC) src/evrard.cpp -o $(BINDIR)/$@.app $(LIB)

mpievrard: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICC) $(CFLAGS) $(INC) -DUSE_MPI src/evrard.cpp -o $(BINDIR)/$@.app $(LIB)

sqpatch: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CC) $(CFLAGS) $(INC) src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

mpisqpatch: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICC) $(CFLAGS) $(INC) -DUSE_MPI src/sqpatch.cpp -o $(BINDIR)/$@.app $(LIB)

run: evrard

clean:
	$(info Cleaning...) 
	$(RM) -rf $(BUILDDIR) $(BINDIR)

.PHONY: clean
