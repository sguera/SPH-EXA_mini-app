CC := g++ # This is the main compiler
MPICC := mpic++

# CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
BINDIR := bin
 
#SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
HPP := $(wildcard src/include/*.hpp)
HPP += $(wildcard src/include/tree/*.hpp)

CFLAGS := -std=c++14 -O2 -s -Wall -Wextra -fopenmp -march=native -mtune=native
DEBUG := -D__DEBUG -D_GLIBCXX_DEBUG
INC := -I src/include
LIB := 

all: evrard
	
evrard: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CC) $(CFLAGS) $(INC) src/evrard.cpp -o $(BINDIR)/$@.app $(LIB)

debug:
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CC) $(CFLAGS) $(INC) $(DEBUG) src/evrard.cpp -o $(BINDIR)/$@.app $(LIB)

mpi: $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICC) $(CFLAGS) $(INC) src/evrard.cpp -o $(BINDIR)/$@.app $(LIB)

run: evrard
	$(info Run the default test case: )
	./bin/evrard.app

clean:
	$(info Cleaning...) 
	$(RM) -rf $(BUILDDIR) $(BINDIR)

# Tests
# tester:
# 	$(CC) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester.app

.PHONY: clean
