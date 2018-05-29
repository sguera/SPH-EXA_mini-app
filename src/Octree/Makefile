CC := g++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src 
BUILDDIR := build
TARGET := runner
CXX := $(CC)
 
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -g -std=c++17 -O2 #-march=native -mtune=native -fopenmp# -Wall
LIB := -L lib #-lpthread #-lboost_thread-mt -lboost_filesystem-mt -lboost_system-mt
INC := -I include
CXXFLAGS := $(CFLAGS)

$(TARGET): $(OBJECTS)
	$(info )
	$(info Linking the executable:)
	$(CC) $(CFLAGS) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	$(info )
	@mkdir -p $(BUILDDIR)
	$(info Compiling the object files:)
	$(CC) $(CFLAGS)  $(LIB) $(INC) -c $< -o $@

clean:
	$(info )
	$(info  Cleaning...) 
	$(RM) -r $(BUILDDIR) $(TARGET)

run: 
	$(info )
	$(info Run the default test case: )
	./runner 

# Tests
# tester:
# 	$(CC) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester

# Spikes
#ticket:
#  $(CC) $(CFLAGS) spikes/ticket.cpp $(INC) $(LIB) -o bin/ticket

.PHONY: clean
