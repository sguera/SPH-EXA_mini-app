#!/bin/bash

# https://jenkins.cscs.ch/view/pasc/

module rm xalt
todo=$1
mkdir -p $SCRATCH/ci
RRR="'-Cmc --wait' null 'cd $SCRATCH/ci;' '' '' '$SCRATCH/ci/'"
#echo "$RRR"
#echo "xxx $RRR"
#exit 0

# USAGE : arg1=machine arg2=walltime arg3=exe
#      arg4=ntasks=-n
#      arg5=ntasks-per-node
#      arg6=cpus-per-task=-c
#      arg7=openmp
#      arg8=ntasks-per-core=-j
#      arg9=sbatchflags
#      arg10=args
#      arg11=preaprun arg12=postaprun
#      arg13=outputpostfix
#      arg14=outputpath
#      arg15=cpubindflag

function git_ci_branch() {
    # should be 01b260c5fe49722eee277b20cfb64219eeabca7d
    echo "pwd=`pwd`"
    git branch -a
    git log -n1
    echo
}

#{{{ # GNU:
 ####   #    #  #    #
#    #  ##   #  #    #
#       # #  #  #    #
#  ###  #  # #  #    #
#    #  #   ##  #    #
 ####   #    #   ####
function compile_and_run_gnu620() {
    module swap PrgEnv-cray PrgEnv-gnu
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
}

function compile_and_run_gnu730() {
    module swap PrgEnv-cray PrgEnv-gnu
    module swap gcc/7.3.0
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
}

function compile_and_run_gnu830() {
    module swap PrgEnv-cray PrgEnv-gnu
    module swap gcc/8.3.0
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
}
 
#old # TODO: /apps/common/UES/sandbox/jgp/production.git/easybuild/easyconfigs/g/GCC/73
#old function compile_and_run_HH() {
#old     # GNU/7.3.0 + OpenMP/4.5
#old     module swap PrgEnv-cray PrgEnv-gnu
#old     module swap gcc/7.3.0
#old     GCCROOT=/apps/common/UES/sandbox/jgp/production.git/easybuild/easyconfigs/g/GCC/73
#old     export PATH=$GCCROOT/bin:$PATH
#old     export LD_LIBRARY_PATH=$GCCROOT/lib64:$LD_LIBRARY_PATH
#old     module list -t
#old     CC --version
#old     make distclean -f Makefile.cscs
#old     make SRC=src/sqpatch.cpp -f Makefile.cscs
#old     sbatchjg
#old }
#}}} 

#{{{ # INTEL: 
#*#   #     # ####### ####### #
 #    ##    #    #    #       #
 #    # #   #    #    #       #
 #    #  #  #    #    #####   #
 #    #   # #    #    #       #
 #    #    ##    #    #       #
###   #     #    #    ####### #######
function compile_and_run_intel17() {
    module swap PrgEnv-cray PrgEnv-intel
    module swap intel/17.0.4.196
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
}

function compile_and_run_intel18() {
    module swap PrgEnv-cray PrgEnv-intel
    module swap intel/18.0.2.199
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
}

function compile_and_run_intel19() {
    module swap PrgEnv-cray PrgEnv-intel
    module swap intel/19.0.1.144
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
}
#}}} 

#{{{ # CCE:
 #####   #####  #######
#     # #     # #
#       #       #
#       #       #####
#       #       #
#     # #     # #
 #####   #####  #######
function compile_and_run_cce86() {
    # module swap PrgEnv-cray PrgEnv-intel
    module swap cce/8.6.1
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
}

function compile_and_run_cce87() {
    # module swap PrgEnv-cray PrgEnv-intel
    module swap cce/8.7.4
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
}

function compile_and_run_cce8710() {
    # module swap PrgEnv-cray PrgEnv-intel
    module swap cce/8.7.10
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
}

#}}}

#{{{ # PGI:
#####    ####      #
#    #  #    #     #
#    #  #          #
#####   #  ###     #
#       #    #     #
#        ####      #
function compile_and_run_pgi1810() {
    module swap PrgEnv-cray PrgEnv-pgi
    module swap pgi/18.10.0
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
    # pgi/18.7.0   
    #echo 'PGCC-S-0101-Illegal operand types for + operator (./src/include/EnergyConservation.hpp: 30)'
}

function compile_and_run_pgi18() {
    module swap PrgEnv-cray PrgEnv-pgi
    module swap pgi/18.5.0
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs # && sbatchjg
    # pgi/18.7.0   
    #echo 'PGCC-S-0101-Illegal operand types for + operator (./src/include/EnergyConservation.hpp: 30)'
}

#}}}

#{{{ # CLANG:
 ####   #         ##    #    #   ####
#    #  #        #  #   ##   #  #    #
#       #       #    #  # #  #  #
#       #       ######  #  # #  #  ###
#    #  #       #    #  #   ##  #    #
 ####   ######  #    #  #    #   ####
function compile_and_run_II() {
    # clang+llvm/7.0.0
    module swap PrgEnv-cray PrgEnv-gnu
    module use /project/c16/easybuild/modules/all
    module load clang+llvm/7.0.0-x86_64-linux-sles12.3
    export PE_ENV=CLANG
    module list -t
    clang++ --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs
    sbatchjg
}
#}}} 

#{{{ # sbatch
function sbatchjg() {
    module list -t
    # git_ci_branch
    RUNDIR=$SCRATCH/ci/$PE_ENV
    rm -fr $RUNDIR/*
    # ./scripts/ci/sbatch.sh -help
    # --- OMP_PLACES={sockets,threads}:
    #for ompp in sockets threads ; do
    for ompp in threads ; do
        echo openmp=$ompp
        mkdir -p $RUNDIR/$ompp
        rm -f $RUNDIR/$ompp/effo_*
        ./scripts/ci/sbatch.sh dom 5 $PWD/bin/*exe 1 1 36 36 1 "-dsingleton -Cmc --wait" noarg "cd $RUNDIR/$ompp;cp -a /project/c16/ci/sph-exa_mini-app.git/bigfiles/ .;OMP_PLACES=$ompp " "" $ompp "$RUNDIR/$ompp/"
        echo -e "\n\n\n\n--- job output ---"
        cat $RUNDIR/$ompp/effo_*$ompp
        grep -q '=== Total time for iteration' $RUNDIR/$ompp/effo_*$ompp;rc=$?
        echo "rc=$rc"
        if [ "$rc" != 0 ]; then
            exit -1
        fi
    done

#slower     # --- default:
#slower     ompp=default
#slower     echo openmp=$ompp
#slower     mkdir -p $SCRATCH/ci/$ompp
#slower     ./scripts/ci/sbatch.sh dom 5 $PWD/bin/*exe 1 1 36 36 1 "-dsingleton -Cmc --wait" noarg "cd $SCRATCH/ci/$ompp;cp -u /project/c16/ci/sph-exa_mini-app.git/bigfiles/ .;" "" $ompp "$SCRATCH/ci/"
#slower     cat $SCRATCH/ci/effo_*$ompp


	# USAGE : arg1=machine arg2=walltime arg3=exe
	#      arg4=ntasks=-n
	#      arg5=ntasks-per-node
	#      arg6=cpus-per-task=-c
	#      arg7=openmp
	#      arg8=ntasks-per-core=-j
	#      arg9=sbatchflags
	#      arg10=args
	#      arg11=preaprun arg12=postaprun
	#      arg13=outputpostfix
	#      arg14=outputpath
	#      arg15=cpubindflag
}
#}}}

function starts_here() {
    module load daint-mc
    echo "todo=$todo CRAY_CPU_TARGET=$CRAY_CPU_TARGET"
    git_ci_branch
    compile_and_run_$todo
}

# ---------------
starts_here $todo
# ---------------

exit 0
