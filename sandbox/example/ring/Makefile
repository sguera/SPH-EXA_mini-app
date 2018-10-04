RM := rm -fr
CXX=g++

CXXFLAGS  = -DARB_PROFILE_ENABLED -DARB_HAVE_PROFILING 
CXXFLAGS += -Iinclude -Iarbor -Iarbor/profile -Iarbor/hardware 
CXXFLAGS += -Iarbor/threading -Iarbor/util

ifeq ($(PE_ENV),GNU)
	CXX=g++ #CC
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
	CXXFLAGS += -std=c++17 -g
	#CXXFLAGS += -std=c++14 -g
	PE_ENV = CLANG
endif

LDFLAGS = -lpthread

EXECUTABLE := $(PE_ENV)_$(COMPILER_VERSION).exe

all: $(EXECUTABLE)

OBJS := execution_context.o \
gpu_context.o \
local_alloc.o \
affinity.o \
memory.o \
node_info.o \
power.o \
clock.o \
memory_meter.o \
meter_manager.o \
power_meter.o \
profiler.o \
threading.o \
thread_info.o \
hostname.o \
ring.o

$(OBJS):
	$(PREP) $(CXX) $(CXXFLAGS) -c src/$(@:.o=.cpp)

$(EXECUTABLE): $(OBJS)
	$(PREP) $(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $(@) 

clean:
	-$(RM) $(OBJS) *.exe *.gch

