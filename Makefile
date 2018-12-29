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

TESTCASE=evrard

all: $(TESTCASE)
	
$(TESTCASE): $(HPP)
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CC) $(CFLAGS) $(INC) src/$(TESTCASE).cpp -o $(BINDIR)/$@.app $(LIB)

debug:
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CC) $(CFLAGS) $(INC) $(DEBUG) src/$(TESTCASE).cpp -o $(BINDIR)/$@.app $(LIB)

run: evrard

clean:
	$(info Cleaning...) 
	$(RM) -rf $(BUILDDIR) $(BINDIR)

.PHONY: clean
