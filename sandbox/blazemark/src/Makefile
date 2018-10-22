# module swap PrgEnv-cray PrgEnv-gnu
# module swap gcc gcc/7.3.0
# module use /apps/dom/UES/sandbox/piccinal/6.0.UP07/gpu/easybuild/modules/all
# module load Boost/1.68.0-CrayGNU-18.10

# Currently Loaded Modulefiles:
#	modules/3.2.10.6
#	eproxy/2.0.22-6.0.7.0_37.1__g1ebe45c.ari
#	gcc/7.3.0
#	craype-haswell
#	craype-network-aries
#	craype/2.5.15
#	cray-mpich/7.7.2
#	slurm/17.11.9.cscs-1
#	cray-libsci/18.07.1
#	udreg/2.3.2-6.0.7.0_33.18__g5196236.ari
#	ugni/6.0.14.0-6.0.7.0_23.1__gea11d3d.ari
#	pmi/5.0.14
#	dmapp/7.1.1-6.0.7.0_34.3__g5a674e0.ari
#	gni-headers/5.0.12.0-6.0.7.0_24.1__g3b1768f.ari
#	xpmem/2.2.15-6.0.7.1_5.8__g7549d06.ari
#	job/2.2.3-6.0.7.0_44.1__g6c4e934.ari
#	dvs/2.7_2.2.112-6.0.7.1_6.4__ge96a422
#	alps/6.6.43-6.0.7.0_26.4__ga796da3.ari
#	rca/2.2.18-6.0.7.0_33.3__g2aa4f39.ari
#	atp/2.1.2
#	perftools-base/7.0.2
#	PrgEnv-gnu/6.0.4
#	daint-gpu
#	CrayGNU/.18.10
#	bzip2/.1.0.6-CrayGNU-18.10
#	zlib/.1.2.11-CrayGNU-18.10
#	Boost/1.68.0-CrayGNU-18.10

RM := rm -fr
#CXX=g++
CXX=CC

CXXFLAGS  = -I/apps/common/UES/sandbox/jgp/production.git/easybuild/easyconfigs/b/blaze/blaze-3.4/
CXXFLAGS += -I/apps/common/UES/sandbox/jgp/production.git/easybuild/easyconfigs/b/blaze/blaze-3.4/blazemark
# CXXFLAGS += -DINSTALL_PATH="'./'"

ifeq ($(PE_ENV),GNU)
	CXX=CC
	COMPILER_VERSION = $(GNU_VERSION)
	CXXFLAGS += -std=c++14 -O3
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

LDFLAGS = -lboost_system -lboost_thread

EXECUTABLE := $(PE_ENV)_$(COMPILER_VERSION).exe

all: $(EXECUTABLE)

OBJS := clike_DVecDVecSub.o \
classic_DVecDVecSub.o \
blaze_DVecDVecSub.o \
boost_DVecDVecSub.o \
DVecDVecSub.o

$(OBJS):
	$(PREP) $(CXX) $(CXXFLAGS) -c main/$(@:.o=.cpp)

$(EXECUTABLE): $(OBJS)
	$(PREP) $(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $(@) 

clean:
	-$(RM) $(OBJS) *.exe *.gch

