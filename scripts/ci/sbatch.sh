#!/bin/bash -l
# -l will set modulecmd

# TODO:
#todo     srun -n $SLURM_NTASKS --ntasks-per-node $SLURM_NTASKS_PER_NODE -c $SLURM_CPUS_PER_TASK --ntasks-per-core $SLURM_NTASKS_PER_CORE hostname 
#todo     set -x
#todo     sacct -o JobID,UID,User,Account,AllocCPUS,AllocNodes,NNodes,NTasks,Elapsed,Start,End,ExitCode,JobName,NodeList,Partition,State,Timelimit,TotalCPU

if [ $# -lt 9 ] ; then 
   echo "USAGE : arg1=machine arg2=walltime arg3=exe "
   echo "     arg4=ntasks=-n" 
   echo "     arg5=ntasks-per-node"
   echo "     arg6=cpus-per-task=-c"
   echo "     arg7=openmp"
   echo "     arg8=ntasks-per-core=-j"
   echo "     arg9=sbatchflags"
   echo "     arg10=args"
   echo "     arg11=preaprun arg12=postaprun"
   echo "     arg13=outputpostfix"
   echo "     arg14=outputpath"
   echo "     arg15=cpubindflag"
   exit 0
fi

function ceiling() {

    cn=`echo $1 $2 |awk '{print ($1/$2)}'`
    awk -vcn=$1 'function ceiling(x){return (x == int(x)) ? x : int(x)+1 }
    BEGIN{ print ceiling(cn) }'

    #awk -vnumber="$numtasks" -vdiv="$mppnppn" '
    #function ceiling(x){return (x == int(x)) ? x : int(x)+1 }
    #BEGIN{ print ceiling(number/div) }'

}

cmd=`echo $0 "$@"`
cluster="$1"
if [ -z $cluster ] ; then
        cluster=`hostname |tr -d [0-9]`
        goto=
else
        if [ ! $cluster = "brisi" ] ; then
                #goto="--clusters=$cluster"
                goto="--clusters=$cluster"
        else 
                goto=
        fi
fi

T="$2"
exe="$3"
mppwidth="$4"
mppnppn="$5"  # "${5:-$cpcn}"

# if [ $cluster = "rothorn" ] ; then
#         cpcn=256
#         mppnppn="${5:-$cpcn}"
# 
# elif [ $cluster = "pilatus" ] ; then
#         cpcn=32
#         mppnppn="${5:-$cpcn}"
# 
# elif [ $cluster = "monch" ] ; then
#         cpcn=40
#         mppnppn="${5:-$cpcn}"
# 
# elif [ $cluster = "escha" ] || [ $cluster = "kesch" ] ; then
#         cpcn=24
#         mppnppn="${5:-$cpcn}"
# 
# else 
# # [ $cluster != "rothorn" ] && [ $cluster != "pilatus" ] && [ $cluster != "monch" ] ; then
#         cpcn=`xtprocadmin -A |cut -c1-85| grep -m 1 compute|awk '{print $7}'`
#         mppnppn="${5:-$cpcn}"
#         if [ $mppnppn -gt $mppwidth ] ; then mppnppn=$mppwidth ; fi
#         #if [ $mppnppn -lt $cpcn ] ; then mppnppn=$mppwidth ; fi
# fi
# # fi

mppdepth="$6"   # "${6:-1}"
openmpth="$7"
if [ "$openmpth" = "0" ] ;then
    ompth="# export OMP_NUM_THREADS="
else
    ompth="export OMP_NUM_THREADS=$openmpth"
fi
# if [ -z $mppdepth ] ;then mppdepth=1 ; fi
hyperthreading="$8"

# if [ [ $cluster = "daint" ]|| [ $cluster = "dom" ] ] ; then
ht1="#SBATCH --ntasks-per-core=$hyperthreading      # -j"
# ht2="--hint=nomultithread"
if [ "$hyperthreading" = 1 ] ;then
    ht2="#SBATCH --hint=nomultithread"
    ht3="--hint=nomultithread"
    ht4="htno"
else
    ht2="#SBATCH --hint=multithread"
    ht3="--hint=multithread"
    ht4="hton"
fi
# fi

# numtasks=`expr $mppwidth \* $mppdepth | xargs printf "%04d\n"`
numtask=`echo $mppwidth | xargs printf "%d\n"`
sbatchflags="$9"
argsexe="${10}"
preaprun="${11}"
postaprun="${12}"
postfix="${13}"
postpath="${14}"
binding="${15}"

if [ -z $binding ] ;then
    cpubd=verbose
else
    cpubd="$binding"
fi

# echo "$mppwidth/$mppnppn"
cnodes=`perl -e "use POSIX qw(ceil);printf \"%d\n\",ceil($mppwidth/$mppnppn)"`
# ==========================> cnodes=`ceiling`

oexe=`basename $exe`
out=eff_runme.slurm.$cluster

 #####  ######     #    #     #
#     # #     #   # #    #   #
#       #     #  #   #    # #
#       ######  #     #    #
#       #   #   #######    #
#     # #    #  #     #    #
 #####  #     # #     #    #
if [ $cluster = "tiger" ] || \
   [ $cluster = "dmi" ] || \
   [ $cluster = "edison" ] || \
   [ $cluster = "daint" ] || \
   [ $cluster = "dom" ] ; then

# echo "CLUSTER=$cluster ..."
cat <<EOF > $out
#!/bin/bash
##SBATCH --nodes=$cnodes
#
#SBATCH --exclusive
#SBATCH --ntasks=$numtask               # -n
#SBATCH --ntasks-per-node=$mppnppn      # -N
#SBATCH --cpus-per-task=$mppdepth       # -d/-c openmp
$ht1
$ht2
#
#SBATCH --time=00:$T:00
#SBATCH --job-name="$PE_ENV"
#SBATCH --output=${postpath}effo_$oexe.$4.$5.$6.$7.$8.$ht4-$cluster.$postfix
#SBATCH --error=${postpath}effo_$oexe.$4.$5.$6.$7.$8.$ht4-$cluster.$postfix
# ---
##SBATCH --constraint=bigmem
##SBATCH --gres=gpu:1     # required for MPS cuda proxy to work
##SBATCH --account=usup
##SBATCH --reservation=maint

echo '# -----------------------------------------------'
ulimit -c unlimited
ulimit -s unlimited
ulimit -a |awk '{print "# "\$0}'
$ompth
export CRAY_OMP_CHECK_AFFINITY=TRUE
export MALLOC_MMAP_MAX_=0
export MALLOC_TRIM_THRESHOLD_=536870912
export MPICH_VERSION_DISPLAY=1
echo '# -----------------------------------------------'

echo '# -----------------------------------------------'
echo "# SLURM_JOB_NODELIST = \$SLURM_JOB_NODELIST"
echo "# SLURM_JOB_NUM_NODES = \$SLURM_JOB_NUM_NODES"
echo "# SLURM_JOB_ID = \$SLURM_JOB_ID"
echo "# SLURM_JOBID = \$SLURM_JOBID"
echo "# SLURM_NTASKS = \$SLURM_NTASKS / -n --ntasks"
echo "# SLURM_NTASKS_PER_NODE = \$SLURM_NTASKS_PER_NODE / -N --ntasks-per-node"
echo "# SLURM_CPUS_PER_TASK = \$SLURM_CPUS_PER_TASK / -d-c --cpus-per-task"
echo "# OMP_NUM_THREADS = \$OMP_NUM_THREADS / -d-c "
echo "# SLURM_NTASKS_PER_CORE = \$SLURM_NTASKS_PER_CORE / -j1 --ntasks-per-core"
# sacct --format=JobID,NodeList%100 -j \$SLURM_JOB_ID
echo '# -----------------------------------------------'
echo '# -----------------------------------------------'
echo "# SLURM_CPUS_ON_NODE = \$SLURM_CPUS_ON_NODE"
echo "# SLURM_LOCALID = \$SLURM_LOCALID"
echo "# SLURM_NNODES = \$SLURM_NNODES"
echo "# SLURM_NODEID = \$SLURM_NODEID"
echo "# SLURM_PROCID = \$SLURM_PROCID"
echo "# SLURM_NPROCS = \$SLURM_NPROCS"
echo "# SLURM_OVERCOMMIT = \$SLURM_OVERCOMMIT"
echo "# nodeid:\$SLURM_NODEID taskid:\$SLURM_PROCID localid:\$SLURM_LOCALID"
echo "# "
echo '# -----------------------------------------------'


date
echo "warning: no --cpu_bind=rank"
set -x
echo CRAY_CUDA_MPS=\$CRAY_CUDA_MPS
echo HUGETLB_DEFAULT_PAGE_SIZE=\$HUGETLB_DEFAULT_PAGE_SIZE
echo HUGETLB_MORECORE=\$HUGETLB_MORECORE
export TMPDIR=/tmp
echo TMPDIR=\$TMPDIR

# ddt --connect \\
# /usr/bin/time -p \\

$preaprun srun \\
--unbuffered \\
--ntasks=\$SLURM_NTASKS \\
--ntasks-per-node=\$SLURM_NTASKS_PER_NODE \\
--cpus-per-task=\$SLURM_CPUS_PER_TASK \\
--ntasks-per-core=\$SLURM_NTASKS_PER_CORE \\
--cpu_bind=$cpubd \\
$ht3 \\
$postaprun $exe $argsexe
set +x
exit 0

--bcast=/tmp/`basename $exe` \\
--cpu_bind=rank \\






echo "# submit command : \"$cmd\""
# grep srun $out
# -N $mppnppn -d $mppdepth $ht2 $postaprun $exe $argsexe 
# grep "SLURM_NTASKS|SLURM_NTASKS_PER_NODE|SLURM_CPUS_PER_TASK|SLURM_NTASKS_PER_CORE|^rank" o
# /usr/bin/time -p $preaprun aprun -n \$SLURM_NTASKS -N \$SLURM_NTASKS_PER_NODE -d \$SLURM_CPUS_PER_TASK $postaprun $exe $argsexe
# export MPICH_CPUMASK_DISPLAY=1        # = core of the rank
# export MPICH_RANK_REORDER_DISPLAY=1   # = node of the rank
# export MPICH_ENV_DISPLAY=1
# export PAT_RT_CALLSTACK_BUFFER_SIZE=50000000 # > 4194312
# export OMP_STACKSIZE=500M
# export PAT_RT_EXPFILE_MAX=99999
# export PAT_RT_SUMMARY=0
# export PAT_RT_TRACE_FUNCTION_MAX=1024 
# export PAT_RT_EXPFILE_PES
# export MPICH_PTL_MATCH_OFF=1
# export MPICH_PTL_OTHER_EVENTS=4096
# export MPICH_MAX_SHORT_MSG_SIZE=32000
# export MPICH_PTL_UNEX_EVENTS=180000
# export MPICH_UNEX_BUFFER_SIZE=284914560
# export MPICH_COLL_OPT_OFF=mpi_allgather
# export MPICH_COLL_OPT_OFF=mpi_allgatherv
# export MPICH_NO_BUFFER_ALIAS_CHECK=1
# export MPICH_MPIIO_STATS=1

EOF

fi
















#     #  #####  #     #
##   ## #     # #     #
# # # # #       #     #
#  #  # #       #######
#     # #       #     #
#     # #     # #     #
#     #  #####  #     #
if [ $cluster = "escha" ] || [ $cluster = "kesch" ] ; then
cat <<EOF > $out
#!/bin/bash
##SBATCH --nodes=$mppnppn
#SBATCH --exclusive
#
#SBATCH --ntasks=$numtask          # -n
#SBATCH --ntasks-per-node=$mppnppn # -N not same as -N
#SBATCH --cpus-per-task=$mppdepth  # -d
###$ht1    # --ntasks-per-core / -j"
###SBATCH --cpu_bind=verbose         # --cpu_bind=v 
#
#SBATCH --time=00:$T:00
#SBATCH --job-name="staff"
#SBATCH --output=o_$oexe.$numtasks.$mppnppn.$mppdepth.$hyperthreading.$cluster.$postfix
#SBATCH --error=o_$oexe.$numtasks.$mppnppn.$mppdepth.$hyperthreading.$cluster.$postfix
##SBATCH --account=usup
##SBATCH --reservation=maint

echo '# -----------------------------------------------'
ulimit -c unlimited
ulimit -s unlimited
ulimit -a |awk '{print "# "\$0}'
echo '# -----------------------------------------------'

echo '# -----------------------------------------------'
export OMP_NUM_THREADS=$mppdepth
#export OMP_STACKSIZE=500M
export MV2_USE_CUDA=0
echo '# -----------------------------------------------'

echo '# -----------------------------------------------'
# export MPICH_CPUMASK_DISPLAY=1        # = core of the rank
# The distribution of MPI tasks on the nodes can be written to the standard output file by setting environment variable 
export MPICH_VERSION_DISPLAY=1
export MPICH_ENV_DISPLAY=1
#export MPICH_PTL_MATCH_OFF=1
#export MPICH_PTL_OTHER_EVENTS=4096
#export MPICH_MAX_SHORT_MSG_SIZE=32000
#export MPICH_PTL_UNEX_EVENTS=180000
#export MPICH_UNEX_BUFFER_SIZE=284914560
#export MPICH_COLL_OPT_OFF=mpi_allgather
#export MPICH_COLL_OPT_OFF=mpi_allgatherv
#export MPICH_NO_BUFFER_ALIAS_CHECK=1
#NEW export MPICH_MPIIO_STATS=1
echo '# -----------------------------------------------'


echo '# -----------------------------------------------'
echo "# SLURM_JOB_NODELIST = \$SLURM_JOB_NODELIST"
echo "# submit command : \"$cmd\""
grep srun $out
echo "# SLURM_JOB_NUM_NODES = \$SLURM_JOB_NUM_NODES"
echo "# SLURM_JOB_ID = \$SLURM_JOB_ID"
echo "# SLURM_JOBID = \$SLURM_JOBID"
echo "# SLURM_NTASKS = \$SLURM_NTASKS / -n --ntasks"
echo "# SLURM_NTASKS_PER_NODE = \$SLURM_NTASKS_PER_NODE / -N --ntasks-per-node"
echo "# SLURM_CPUS_PER_TASK = \$SLURM_CPUS_PER_TASK / -d --cpus-per-task"
echo "# OMP_NUM_THREADS = \$OMP_NUM_THREADS / -d "
echo "# SLURM_NTASKS_PER_CORE = \$SLURM_NTASKS_PER_CORE / -j1 --ntasks-per-core"
# sacct --format=JobID,NodeList%100 -j \$SLURM_JOB_ID
echo '# -----------------------------------------------'

echo '# -----------------------------------------------'
# ldd $exe |grep intel 2> /dev/null
# if [ \$? -eq 0 ] ; then 
#         # intel executables
#         export KMP_AFFINITY=disabled 
# else
#         export KMP_AFFINITY=enabled
#         #export GOMP_CPU_AFFINITY="\$cpuaff"
# fi
echo '# -----------------------------------------------'

date
set +x

export KMP_AFFINITY=verbose,auto
export G2G=1
/usr/bin/time -p $preaprun mpiexec.hydra -print-rank-map -rmk slurm $postaprun $exe $argsexe # warning: srun != -n$mppnppn -N$mppwidth -d$mppdepth ok

#ok /usr/bin/time -p $preaprun Srun --cpu_bind=verbose -N $mppnppn -n $mppwidth -c $mppdepth $postaprun $exe $argsexe
#/usr/bin/time -p $preaprun Srun --cpu_bind=verbose -n \$SLURM_NTASKS -N \$SLURM_NTASKS_PER_NODE -d \$SLURM_CPUS_PER_TASK $ht2 $postaprun $exe $argsexe

# isintelmpi=`which mpiexec |grep -q intel ; echo $?`
# if [ \$isintelmpi = 0 ] ; then 
#        time -p $preaprun Mpirun  -rmk slurm                       $postaprun $exe $argsexe
# fi
# /usr/bin/time -p $preaprun Aprun -n $mppwidth -N $mppnppn -d $mppdepth $ht2 $postaprun $exe $argsexe 
# mv wave_tank1.h5 $mppwidth.wave_tank1.h5
EOF
### egrep "SLURM_NTASKS|SLURM_NTASKS_PER_NODE|SLURM_CPUS_PER_TASK" $out |grep echo
fi



 #####   #####    #*#
#     # #     #    #
#       #          #
 #####  #  ####    #
      # #     #    #
#     # #     #    #
 #####   #####    ###
#sgi if [ $cluster = "rothorn" ] ; then
#sgi 
#sgi if [ ! -x /users/piccinal/sgi2.sh ] ; then
#sgi         echo "/users/piccinal/sgi2.sh is missing"
#sgi         exit -1
#sgi fi
#sgi 
#sgi memsgi=`expr $numtasks \* 8000` # GB
#sgi 
#sgi 
#sgi cat <<EOF > $out
#sgi #!/bin/bash
#sgi #SBATCH --account=usup
#sgi #SBATCH --nodes=1 
#sgi #SBATCH --threads-per-core=1
#sgi #SBATCH --cpu_bind=verbose
#sgi #SBATCH --cpus-per-task=$mppdepth
#sgi #SBATCH --ntasks-per-node=$mppwidth
#sgi #SBATCH --time=00:$T:00
#sgi #SBATCH --job-name="jg"
#sgi #SBATCH --output=o_$oexe.$numtasks.$mppnppn.$mppdepth.$cluster
#sgi #SBATCH --error=o_$oexe.$numtasks.$mppnppn.$mppdepth.$cluster
#sgi #SBATCH --mem=$memsgi
#sgi ##SBATCH --mem=1700000 => MB
#sgi # --mem-per-cpu=4096
#sgi # -V
#sgi # --cpus-per-task | mppdepth
#sgi # --ntasks | mppwidth
#sgi #  --ntasks-per-core=<ntasks>
#sgi #  --ntasks-per-socket=<ntasks>
#sgi #  --ntasks-per-node=<ntasks>
#sgi # -----------------------------------------------------------------------
#sgi . /etc/bash.bashrc.local
#sgi . /etc/profile.d/modules.sh
#sgi #module load PrgEnv-intel/12.0.0 hdf5-parallel/1.8.8 slurm
#sgi 
#sgi # print the physical core ID allocated on the system for the current jobID
#sgi # cat /dev/cpuset/$(cpuset -w0)/cpus        
#sgi # cpu_exclusive
#sgi # cpus
#sgi # mem_exclusive
#sgi # mem_hardwall
#sgi # memory_migrate
#sgi # memory_pressure
#sgi # memory_spread_page
#sgi # memory_spread_slab
#sgi # mems
#sgi # notify_on_release
#sgi # sched_load_balance
#sgi # sched_relax_domain_level
#sgi # tasks
#sgi # cgroup.procs
#sgi 
#sgi cpset=\`cpuset -w0\`
#sgi r="/dev/cpuset/\`cpuset -w0\`"
#sgi echo "cpuset -w0 : \$r"
#sgi #echo "cpuset -w0 : /dev/cpuset/\`cpuset -w0\`"
#sgi #echo "== CPUs:"         ; cat /dev/cpuset/`cpuset -w0`/cpus
#sgi echo "==> Job size: \`cpuset -z \$cpset \` "  
#sgi #echo "==> Job size:"     ; cpuset -z `cpuset -w 0`
#sgi 
#sgi echo "==> CPU mapping:"  ; dplace -q
#sgi #echo "== PID list:"     ; cpuset -p `cpuset -w 0`
#sgi for i in cpus cpu_exclusive mem_exclusive mems ;do
#sgi         echo "==> \$i . \`cat \$r/\$i\` ."
#sgi done
#sgi 
#sgi dplaceaff=\`dplace -q |grep -v LCPU |awk '{printf "%d,",\$1}'\`
#sgi echo "dplaceaff=\$dplaceaff"
#sgi cpuaff=\`dplace -q |grep -v PCPU |awk '{printf "%d ",\$2}'\`
#sgi # export GOMP_CPU_AFFINITY="\$cpuaff"
#sgi 
#sgi pwd
#sgi unset mc
#sgi ulimit -s unlimited
#sgi 
#sgi # -----------------------------------------------------------------------
#sgi # echo  "Running on nodes $SLURM_JOB_NODELIST"
#sgi export MPICH_CPUMASK_DISPLAY=1
#sgi # export MALLOC_MMAP_MAX_=0
#sgi # export MALLOC_TRIM_THRESHOLD_=536870912
#sgi export OMP_NUM_THREADS=$mppdepth
#sgi export MPICH_VERSION_DISPLAY=1
#sgi export MPICH_ENV_DISPLAY=1
#sgi #export PAT_RT_CALLSTACK_BUFFER_SIZE=50000000 # > 4194312
#sgi 
#sgi # export PAT_RT_EXPFILE_MAX=99999
#sgi # export PAT_RT_SUMMARY=0
#sgi #
#sgi #export PAT_RT_TRACE_FUNCTION_MAX=1024 
#sgi #export PAT_RT_EXPFILE_PES
#sgi #export MPICH_PTL_MATCH_OFF=1
#sgi #export MPICH_PTL_OTHER_EVENTS=4096
#sgi #export MPICH_MAX_SHORT_MSG_SIZE=32000
#sgi #export MPICH_PTL_UNEX_EVENTS=180000
#sgi #export MPICH_UNEX_BUFFER_SIZE=284914560
#sgi 
#sgi ldd $exe |grep intel 2> /dev/null
#sgi if [ \$? -eq 0 ] ; then 
#sgi         # intel executables
#sgi         export KMP_AFFINITY=disabled 
#sgi else
#sgi         export KMP_AFFINITY=enabled
#sgi         #export GOMP_CPU_AFFINITY="\$cpuaff"
#sgi fi
#sgi echo "SLURM_JOB_NAME=\$SLURM_JOB_NAME" "SLURM_JOBID=\$SLURM_JOBID SLURM_JOB_ID=\$SLURM_JOB_ID SLURM_TASK_PID=\$SLURM_TASK_PID OMP_NUM_THREADS=\$OMP_NUM_THREADS KMP_AFFINITY=\$KMP_AFFINITY"
#sgi 
#sgi ldd $exe |grep mpi 2> /dev/null
#sgi if [ \$? -eq 0 ] ; then 
#sgi         # mpi 
#sgi         echo "/usr/bin/time -p $preaprun mpirun -np $mppwidth dplace -o dplace.out -s1 -e -c \$dplaceaff $postaprun $exe $argsexe "
#sgi         /usr/bin/time -p $preaprun mpirun -np $mppwidth dplace -o dplace.out -s1 -e -c \$dplaceaff $postaprun $exe $argsexe 
#sgi         #/usr/bin/time -p $preaprun mpirun -np $mppwidth omplace -nt $mppdepth -vv $postaprun $exe $argsexe 
#sgi else
#sgi         /usr/bin/time -p $preaprun dplace -o dplace.out -s1 -e -c \$dplaceaff $postaprun $exe $argsexe
#sgi fi
#sgi 
#sgi echo "submit command : $cmd"
#sgi #echo "GOMP_CPU_AFFINITY=\$GOMP_CPU_AFFINITY OMP_NUM_THREADS=\$OMP_NUM_THREADS"
#sgi 
#sgi echo "cpu_affinity = \$cpuaff :" 
#sgi /users/piccinal/sgi2.sh "\$cpuaff"
#sgi cat dplace.out ; rm dplace.out
#sgi 
#sgi date +%D:%Hh%Mm%S
#sgi EOF
#sgi fi

######    ###   #          #    ####### #     #  #####
#     #    #    #         # #      #    #     # #     #
#     #    #    #        #   #     #    #     # #
######     #    #       #     #    #    #     #  #####
#          #    #       #######    #    #     #       #
#          #    #       #     #    #    #     # #     #
#         ###   ####### #     #    #     #####   #####
if [ $cluster = "pilatus" ] || [ $cluster = "monch" ] ; then
mempilatus=`expr $numtasks \* 4000` # MB
cat <<EOF > $out
#!/bin/bash
##SBATCH --account=usup
#SBATCH -N $cnodes 
#SBATCH -n $mppwidth
#SBATCH --ntasks-per-node=$mppnppn
#SBATCH --cpu_bind=verbose
#SBATCH --time=00:$T:00
#SBATCH --job-name="jg"
#SBATCH --output=o_$oexe.$numtasks.$mppwidth.$mppnppn.$mppdepth.$cluster
#SBATCH --error=o_$oexe.$numtasks.$mppwidth.$mppnppn.$mppdepth.$cluster
##SBATCH --mem=63000 # MB

##SBATCH --nodes=$cnodes 
##SBATCH --threads-per-core=1
##SBATCH --cpus-per-task=$mppdepth
# -V
# --cpus-per-task | mppdepth
# --ntasks | mppwidth
#  --ntasks-per-core=<ntasks>
#  --ntasks-per-socket=<ntasks>
#  --ntasks-per-node=<ntasks>
# -----------------------------------------------------------------------
pwd
unset mc
ulimit -s unlimited

# -----------------------------------------------------------------------
# export MV2_ENABLE_AFFINITY=NO                                                                                
# *** Intel Only ***
# export KMP_AFFINITY=scatter,verbose
# export KMP_AFFINITY=compact,verbose
# -----------------------------------------------------------------------
# echo  "Running on nodes $SLURM_JOB_NODELIST"
export MPICH_CPUMASK_DISPLAY=1
# export MALLOC_MMAP_MAX_=0
# export MALLOC_TRIM_THRESHOLD_=536870912
export OMP_NUM_THREADS=$mppdepth
export MPICH_VERSION_DISPLAY=1
export MPICH_ENV_DISPLAY=1
#export PAT_RT_CALLSTACK_BUFFER_SIZE=50000000 # > 4194312

# export PAT_RT_EXPFILE_MAX=99999
# export PAT_RT_SUMMARY=0
#
#export PAT_RT_TRACE_FUNCTION_MAX=1024 
#export PAT_RT_EXPFILE_PES
#export MPICH_PTL_MATCH_OFF=1
#export MPICH_PTL_OTHER_EVENTS=4096
#export MPICH_MAX_SHORT_MSG_SIZE=32000
#export MPICH_PTL_UNEX_EVENTS=180000
#export MPICH_UNEX_BUFFER_SIZE=284914560

# ldd $exe |grep intel 2> /dev/null
# if [ \$? -eq 0 ] ; then 
#         # intel executables
#         export KMP_AFFINITY=disabled 
# else
#         export KMP_AFFINITY=enabled
#         #export GOMP_CPU_AFFINITY="\$cpuaff"
# fi
echo "SLURM_JOB_NAME=\$SLURM_JOB_NAME" "SLURM_JOBID=\$SLURM_JOBID SLURM_JOB_ID=\$SLURM_JOB_ID SLURM_TASK_PID=\$SLURM_TASK_PID OMP_NUM_THREADS=\$OMP_NUM_THREADS KMP_AFFINITY=\$KMP_AFFINITY VT_BUFFER_SIZE=\$VT_BUFFER_SIZE VT_MAX_FLUSHES=\$VT_MAX_FLUSHES VT_MODE=\$VT_MODE"

set +x
if [ $cluster = "monch" ] ; then
        isintelmpi=`which mpiexec |grep -q     impi ; echo $?`
         isopenmpi=`which mpiexec |grep -q  openmpi ; echo $?`
         ismvapich=`which mpiexec |grep -q mvapich2 ; echo $?`
        if [ \$isintelmpi = 0 ] ; then time -p $preaprun mpirun  -rmk slurm                       $postaprun $exe $argsexe ; fi
        if [ \$isopenmpi  = 0 ] ; then time -p $preaprun mpiexec -np $mppwidth -npernode $mppnppn $postaprun $exe $argsexe ; fi
        if [ \$ismvapich  = 0 ] ; then time -p $preaprun mpiexec -np $mppwidth -ppn      $mppnppn $postaprun $exe $argsexe ; fi
else
        time -p $preaprun srun $postaprun $exe $argsexe 
fi
#scan get confused : /usr/bin/time -p $preaprun srun --nodes=$cnodes --ntasks=$mppwidth --ntasks-per-node=$mppnppn $postaprun $exe $argsexe 
date +%D:%Hh%Mm%S

# ~/sbatch.sh pilatus ./a.out 1 4 4 8 # = 1min 4mpi*8omp 4mpi/1cn=1cn 
# ~/sbatch.sh pilatus ./a.out 1 4 1 8 # = 1min 4mpi*8omp 1mpi/1cn=4cn 

EOF

fi

# eiger
# srun -n1 --gres=gpu:1 --constraint=gtx285 --pty /bin/bash

 #####  ######     #    #######  #####  #     #
#     # #     #   # #      #    #     # #     #
#       #     #  #   #     #    #       #     #
 #####  ######  #     #    #    #       #######
      # #     # #######    #    #       #     #
#     # #     # #     #    #    #     # #     #
 #####  ######  #     #    #     #####  #     #
# echo sbatch --clusters=$cluster $sbatchflags $out
#if [ -z $cluster ] ; then
#        sbatch                     $sbatchflags $out
#else
if [ $cluster = "dmi" ] || [ $cluster = "tiger" ] || [ $cluster = "edison" ] ;then
	goto=
    #export MV2_ENABLE_AFFINITY=0
    echo "MV2_ENABLE_AFFINITY=$MV2_ENABLE_AFFINITY # !!!"
fi
goto=
jobid=`sbatch $goto $sbatchflags $out | awk '{print $4}'`
echo "Submitted batch job $jobid on cluster $cluster"
if [ ! -z $jobid ] ;then
    scontrol show job $jobid # | grep -A3 NumNodes=
fi
#fi
# grep -E "aprun|mpirun|srun" $out |grep -v echo

exit 0

# xpat
export PAT_RT_HWPC=DATA_CACHE_REFILLS_FROM_L2_OR_NORTHBRIDGE
# scalasca
export EPK_METRICS=DATA_CACHE_REFILLS_FROM_L2_OR_NORTHBRIDGE
