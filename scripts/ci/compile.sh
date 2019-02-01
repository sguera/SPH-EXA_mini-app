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
    # should be 40a7153e460146cc53704b05db561a9c5a6992f8
    echo "pwd=`pwd`"
    git branch -a
    git log -n3
}

#{{{ # sbatch
function sbatchjg() {
    module list -t
    git_ci_branch
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

#{{{ # GNU:
 ####   #    #  #    #
#    #  ##   #  #    #
#       # #  #  #    #
#  ###  #  # #  #    #
#    #  #   ##  #    #
 ####   #    #   ####
function compile_and_run_AA() {
    # GNU/6.2.0
    module swap PrgEnv-cray PrgEnv-gnu
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs
    sbatchjg
}

function compile_and_run_BB() {
    # GNU/7.3.0
    module swap PrgEnv-cray PrgEnv-gnu
    module swap gcc/7.3.0
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs
    sbatchjg
}
 
# TODO: /apps/common/UES/sandbox/jgp/production.git/easybuild/easyconfigs/g/GCC/73
function compile_and_run_HH() {
    # GNU/7.3.0 + OpenMP/4.5
    module swap PrgEnv-cray PrgEnv-gnu
    module swap gcc/7.3.0
    GCCROOT=/apps/common/UES/sandbox/jgp/production.git/easybuild/easyconfigs/g/GCC/73
    export PATH=$GCCROOT/bin:$PATH
    export LD_LIBRARY_PATH=$GCCROOT/lib64:$LD_LIBRARY_PATH
    module list -t
    CC --version
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs
    sbatchjg
}
#}}} 

#{{{ # INTEL: 
#*#   #     # ####### ####### #
 #    ##    #    #    #       #
 #    # #   #    #    #       #
 #    #  #  #    #    #####   #
 #    #   # #    #    #       #
 #    #    ##    #    #       #
###   #     #    #    ####### #######
function compile_and_run_CC() {
    # intel/18.0.2.199
    module swap PrgEnv-cray PrgEnv-intel
    module list -t
    CC -V
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs
    sbatchjg
    #echo "./sbatch.sh dom 5 $PWD/bin/*exe 1 1 36 36 1 -Cmc $RRR"
}

function compile_and_run_DD() {
    # intel/19.0.1.144
    module swap PrgEnv-cray PrgEnv-intel
    module swap intel/19.0.1.144
    module list -t
    CC -V
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs
    sbatchjg
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
function compile_and_run_EE() {
    # cce/8.7.6
    #echo "CCE -homp = INTERNAL COMPILER ERROR, exiting"
    #exit 0
    module swap cce/8.7.6
    module list -t
    CC -V
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs
    sbatchjg
}

function compile_and_run_GG() {
    # cce/8.6.1
    #echo "CCE -homp = INTERNAL COMPILER ERROR, exiting"
    #exit 0
    module swap cce/8.6.1
    module list -t
    CC -V
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs
    sbatchjg
}
#}}}

#{{{ # PGI:
#####    ####      #
#    #  #    #     #
#    #  #          #
#####   #  ###     #
#       #    #     #
#        ####      #
function compile_and_run_FF() {
    # pgi/18.7.0   
    #echo 'PGCC-S-0101-Illegal operand types for + operator (./src/include/EnergyConservation.hpp: 30)'
    #exit 0
    module swap PrgEnv-cray PrgEnv-pgi
    module swap pgi/18.7.0
    module list -t
    CC -V
    make distclean -f Makefile.cscs
    make SRC=src/sqpatch.cpp -f Makefile.cscs
    sbatchjg
}
#}}}

function starts_here() {
    module load daint-mc
    echo "todo=$todo CRAY_CPU_TARGET=$CRAY_CPU_TARGET"
    case $todo in
        AA) compile_and_run_AA;;
        BB) compile_and_run_BB;;
        CC) compile_and_run_CC;;
        DD) compile_and_run_DD;;
        EE) compile_and_run_EE;;
        FF) compile_and_run_FF;;
        GG) compile_and_run_GG;;
        HH) compile_and_run_HH;;
        II) compile_and_run_II;;
        *) echo "todo=$todo is not supported, exiting...";exit -1;;
    esac
}

#function 

starts_here $todo
exit 0

