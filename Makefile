CC := g++ # This is the main compiler
MPICC := mpic++

# CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
BINDIR := bin
 
#SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
HPP := $(wildcard src/*.hpp)
HPP += $(wildcard src/tree/*.hpp)

CFLAGS := -g -std=c++14 -O3 -Wall -Wextra -fopenmp -march=native -mtune=native
LIB := 
INC := -I include

runner: $(HPP)
	$(info )
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CC) $(CFLAGS) src/main.cpp -o $(BINDIR)/$@.app $(LIB)

mpirunner: $(HPP)
	$(info )
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(MPICC) $(CFLAGS) src/distmain.cpp -o $(BINDIR)/$@.app $(LIB)

clean:
	$(info )
	$(info Cleaning...) 
	$(RM) -rf $(BUILDDIR) $(BINDIR)

run: runner
	$(info )
	$(info Run the default test case: )
	./bin/runner.app

mpirun: mpirunner
	$(info )
	$(info Run the default test case: )
	mpirun -N 4 ./bin/runner.app 

# Tests
# tester:
# 	$(CC) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester.app

.PHONY: clean
