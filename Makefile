CC := g++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
BINDIR := bin
TARGET := runner
 
SRCEXT := cpp
#SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
SOURCES := $(wildcard $(SRCDIR)/*.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -g -std=c++14 -O2 -fopenmp -march=native -mtune=native# -Wall
LIB := #-L #lib #-lpthread #-lboost_thread-mt -lboost_filesystem-mt -lboost_system-mt
INC := -I include

$(TARGET): $(OBJECTS)
	$(info )
	@mkdir -p $(BINDIR)
	$(info Linking the executable:)
	$(CC) $(CFLAGS) $^ -o $(BINDIR)/$(TARGET).app $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	$(info )
	@mkdir -p $(BUILDDIR)
	$(info Compiling the object files:)
	$(CC) $(CFLAGS)  $(LIB) $(INC) -c $< -o $@


clean:
	$(info )
	$(info  Cleaning...) 
	$(RM) -rf $(BUILDDIR) $(BINDIR)

run: $(TARGET)
	$(info )
	$(info Run the default test case: )
	./bin/runner.app 

# Tests
# tester:
# 	$(CC) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester.app

.PHONY: clean
