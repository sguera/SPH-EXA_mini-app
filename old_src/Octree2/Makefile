# CXX = clang#g++
# COMPFLAGS = -Wall -Werror -Wextra -pedantic -g -std=c++11 -O2   
# OPTFLAGS = #-march=native -mtune=native
# OMPFLAGS = #-fopenmp
# REPFLAGS = #-qopt-report=5 -qopt-report-filter="GSimulation.cpp,130-220" 
# INCLUDES = 

# CXXFLAGS = $(COMPFLAGS) $(OPTFLAGS) $(REPFLAGS) $(OMPFLAGS)

# HEADER_SUBJECTS = boundingbox.h octree.h point3d.h inneriterator
# SUBJECTS = boundingbox point3d main

# all: cpu

# data_structures/%.o: data_structures/%.cc data_structures/%.h
# 	$(CXX) $(CXX_FLAGS) $< -c -o $@ 

# cpu: $(OBJSC)
# 	$(info )
# 	$(info Linking the CPU executable:)
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXEC) $(OBJSC)
	
# run: 
# 	$(info )
# 	$(info Run the default test case on CPU: )
# 	./main 
	
# clean: 
# 	rm -f $(OBJSC) nbody.x *.optrpt 


CXX = g++
COMPFLAGS = -Wall -pedantic -g -std=c++11 -O2   
OPTFLAGS = -march=native -mtune=native
OMPFLAGS = -fopenmp
REPFLAGS = #-qopt-report=5 -qopt-report-filter="GSimulation.cpp,130-220" 
INCLUDES = 

CXXFLAGS = $(COMPFLAGS) $(OPTFLAGS) $(REPFLAGS) $(OMPFLAGS) 

SOURCES = boundingbox.cc point3d.cc main.cc

.SUFFIXES: .o .cc

##########################################
OBJSC = $(SOURCES:.cpp=.o)
##########################################

EXEC=octree.x

all: cpu

%.o: %.cc
	$(info )
	$(info Compiling the object file for CPU: )
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 

cpu: $(OBJSC)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXEC) $(OBJSC)
	
run: 
	$(info )
	$(info Run the default test case on CPU: )
	./$(EXEC)
	
clean: 
	rm -f $(OBJSC)$(EXEC) *.optrpt 