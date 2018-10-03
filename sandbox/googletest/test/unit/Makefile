RM := rm -fr
CXX=g++

ifeq ($(EBROOTGOOGLETEST),)
$(error EBROOTGOOGLETEST is not set, module load googletest/1.8.1-CrayGNU-18.08)
endif

CXXFLAGS = -I$(EBROOTGOOGLETEST)/include/gtest -I./include -I./arbor

ifeq ($(PE_ENV),GNU)
	CXX=g++
	COMPILER_VERSION = $(GNU_VERSION)
	CXXFLAGS += -std=c++14 -g
endif

ifeq ($(PE_ENV),PGI)
	CXX=pg++
	COMPILER_VERSION = $(PGI_VERSION)
	CXXFLAGS += -std=c++14
endif

ifeq ($(PE_ENV),CRAY)
	CXX=CC
	COMPILER_VERSION = $(CRAY_CC_VERSION)
	CXXFLAGS += -hstd=c++14
endif

ifeq ($(PE_ENV),INTEL)
	CXX=icpc
	COMPILER_VERSION = $(INTEL_VERSION)
	CXXFLAGS += -std=c++14
endif

ifeq ($(CXX),clang++)
	COMPILER_VERSION = $(EBVERSIONCLANGPLUSLLVM)
	CXXFLAGS += -std=c++14 -g
	PE_ENV = CLANG
endif

LDFLAGS = -L$(EBROOTGOOGLETEST)/lib64 -lgtest -lpthread

EXECUTABLE := $(PE_ENV)_$(COMPILER_VERSION).exe

all: $(EXECUTABLE)

OBJS := test_algorithms.o test.o

$(OBJS):
	$(PREP) $(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

$(EXECUTABLE): $(OBJS)
	$(PREP) $(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $(@) 

clean:
	-$(RM) $(OBJS) *.exe *.gch

