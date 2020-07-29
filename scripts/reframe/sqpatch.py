# Copyright 2019-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# HPCTools Project Developers. See the top-level LICENSE file for details.
#
# ~/reframe.git/bin/reframe -C ./cscs.py --system daint:gpu -r -p PrgEnv-gnu -c
# ./test2.py --performance-report -J partition=debug -J account=usup
# --keep-stage-files --prefix=$SCRATCH/aurelien -v
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                'common')))  # noqa: E402
import sphexa.sanity as sphs


# {{{ https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app/issues/34
# ORI:    mpi steps   cube    JG:     mpi steps   cube
# xsmall  1   0       20      xsmall  1   0       30
#  small  4   100     20       small  6   100     30
# medium  4   10      100     medium  24  10      100
# large   16  10      300     large   672 10      300 = 28cn
#
# DEBUG:  mpi steps   cube (weak = >40K particles/core)
# xsmall  1   0       30   30^3/1  = 27 000
#  small  6   1       30   62^3/6  = 39 721
# medium  12  2       78   78^3/12 = 39 546
#
# prg models: mpi+omp, mpi+omp+target, mpi+omp+acc, mpi+omp+cuda
# prg envs: gcc, clang, cray cce, intel and pgi
#
# The test should be a success if mpi+omp and mpi+omp+cuda models are passing.
# size_dict = {12: 78, 24: 100, 48: 125, 96: 157, 192: 198, 384: 250, 480: 269,
#             960: 340, 1920: 428, 3840: 539, 7680: 680, 15360: 857,
#             6: 62, 3: 49, 1: 34}
# }}}
# NOTE: jenkins restricted to 1 cnode
mpi_tasks = [1, 6, 12]
cubeside_dict = {1: 30, 6: 30, 12: 30}
steps_dict = {1: 0, 6: 0, 12: 0}
# cubeside_dict = {1: 30, 6: 62, 12: 78}
# steps_dict = {1: 0, 6: 1, 12: 2}
# nativejob_stdout = 'rfm_native_job.out'


# {{{ class SphExa_MPI
@rfm.parameterized_test(*[[mpi_task]
                          for mpi_task in mpi_tasks
                          ])
class SphExa_MPI_Check(rfm.RegressionTest):
    def __init__(self, mpi_task):
        # super().__init__()
        # {{{ pe
        self.descr = 'Tool validation'
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel',
                                    'PrgEnv-pgi', 'PrgEnv-cray']
        # self.sourcesdir = None
        # self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_systems = ['*']
        self.maintainers = ['JG']
        self.tags = {'sph', 'hpctools', 'cpu'}
        # }}}

        # {{{ compile
        self.testname = 'sqpatch'
        # self.modules = ['atp']
        self.sourcesdir = 'src_cpu'
        self.build_system = 'SingleSource'
        # self.build_system.cxx = 'CC'
        self.sourcepath = '%s.cpp' % self.testname
        prebuild_cmds = [
            'module rm xalt',
        ]
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-I.', '-I./include', '-std=c++14', '-g', '-O2',
                           '-DUSE_MPI', '-DNDEBUG'],
            'PrgEnv-intel': ['-I.', '-I./include', '-std=c++14', '-g', '-O2',
                             '-DUSE_MPI', '-DNDEBUG'],
            'PrgEnv-cray': ['-I.', '-I./include', '-std=c++17', '-g', '-O2',
                            '-DUSE_MPI', '-DNDEBUG'],
            'PrgEnv-pgi': ['-I.', '-I./include', '-std=c++14', '-g', '-O2',
                           '-DUSE_MPI', '-DNDEBUG'],
        }
        # self.executable = self.native_executable
        # }}}

        # {{{ run
        ompthread = 1
        self.num_tasks = mpi_task
        self.num_tasks_per_node = 12
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.num_cpus_per_task = ompthread
        self.exclusive = True
        self.time_limit = '10m'
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        self.cubeside = cubeside_dict[mpi_task]
        self.steps = steps_dict[mpi_task]
        self.executable_opts = [f"-n {self.cubeside}", f"-s {self.steps}"]
        self.prerun_cmds += ['module rm xalt']
        # }}}

        # {{{ sanity
        # self.sanity_patterns_l = [
        self.sanity_patterns = \
            sn.assert_found(r'Total time for iteration\(0\)', self.stdout)
        # self.sanity_patterns = sn.all(self.sanity_patterns_l)
        # }}}

        # {{{ performance
        # {{{ internal timers
        # use linux date as timer:
        self.prerun_cmds += ['echo starttime=`date +%s`']
        self.postrun_cmds += ['echo stoptime=`date +%s`']
        # }}}

        # {{{ perf_patterns:
        self.perf_patterns = sn.evaluate(sphs.basic_perf_patterns(self))
        # }}}

        # {{{ reference:
        self.reference = sn.evaluate(sphs.basic_reference_scoped_d(self))
        # }}}
        # }}}

    # {{{ hooks
    @rfm.run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cxxflags = \
            self.prgenv_flags[self.current_environ.name]
    # }}}
# }}}


# {{{ class SphExa_MPI+CUDA
@rfm.parameterized_test(*[[mpi_task]
                          for mpi_task in mpi_tasks
                          ])
class SphExa_MPI_OpenMP_Cuda_Check(rfm.RegressionTest):
    def __init__(self, mpi_task):
        # {{{ pe
        self.descr = 'Tool validation'
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.sourcesdir = 'src_gpu'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_systems = ['*']
        self.maintainers = ['JG']
        self.tags = {'sph', 'hpctools', 'gpu'}
        # }}}

        # {{{ compile
        self.testname = 'sqpatch'
        self.prebuild_cmds = ['module rm xalt']
        self.modules = ['craype-accel-nvidia60']
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-std=c++14', '-g', '-O2', '-DNDEBUG'],
            # Makefile adds: '-DUSE_MPI', '-DUSE_CUDA'],
        }
        self.prgenv_gpuflags = {
            # P100 = sm_60
            'PrgEnv-gnu': ['-std=c++14', '-rdc=true', '-arch=sm_60',
                           '--expt-relaxed-constexpr']
        }
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile'
        self.executable = 'mpi+omp+cuda'
        self.build_system.options = [
            self.executable, 'MPICXX=CC', 'SRCDIR=.', 'BUILDDIR=.',
            'BINDIR=.', 'CUDA_PATH=$CUDATOOLKIT_HOME',
            # The makefile adds -DUSE_MPI
            # 'CXXFLAGS=',
        ]
        self.sourcesdir = 'src_gpu'
        self.build_system.nvcc = 'nvcc'
        self.build_system.cxx = 'CC'
        self.build_system.max_concurrency = 2
        # }}}

        # {{{ run
        ompthread = 1
        self.num_tasks = mpi_task
        self.num_tasks_per_node = 1
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.num_cpus_per_task = ompthread
        self.exclusive = True
        self.time_limit = '10m'
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        self.cubeside = cubeside_dict[mpi_task]
        self.steps = steps_dict[mpi_task]
        self.executable_opts = [f"-n {self.cubeside}", f"-s {self.steps}"]
        self.prerun_cmds += [
            'module rm xalt',
            f'mv {self.executable}.app {self.executable}'
        ]
        # }}}

        # {{{ sanity
        self.sanity_patterns = \
            sn.assert_found(r'Total time for iteration\(0\)', self.stdout)
        # }}}

        # {{{ performance
        # {{{ internal timers
        # use linux date as timer:
        self.prerun_cmds += ['echo starttime=`date +%s`']
        self.postrun_cmds += ['echo stoptime=`date +%s`']
        # }}}

        # {{{ perf_patterns:
        self.perf_patterns = sn.evaluate(sphs.basic_perf_patterns(self))
        # }}}

        # {{{ reference:
        self.reference = sn.evaluate(sphs.basic_reference_scoped_d(self))
        # }}}
        # }}}

    # {{{ hooks
    @rfm.run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cxxflags = \
            self.prgenv_flags[self.current_environ.name]
    # }}}
# }}}


# WIP # {{{ class SphExa_MPI+OpenMP_TARGET
# WIP @rfm.parameterized_test(*[[mpi_task]
# WIP                           for mpi_task in mpi_tasks
# WIP                           ])
# WIP class SphExa_MPI_OpenMP_Target_Check(rfm.RegressionTest):
# WIP     def __init__(self, mpi_task):
# WIP         # {{{ pe
# WIP         self.descr = 'Tool validation'
# WIP         self.valid_prog_environs = ['PrgEnv-cray']
# WIP         self.sourcesdir = 'src_gpu'
# WIP         self.valid_systems = ['daint:gpu', 'dom:gpu']
# WIP         self.valid_systems = ['*']
# WIP         self.maintainers = ['JG']
# WIP         self.tags = {'sph', 'hpctools', 'gpu'}
# WIP         # }}}
# WIP
# WIP         # {{{ compile
# WIP         self.testname = 'sqpatch'
# WIP         self.prebuild_cmds = ['module rm xalt']
# WIP         self.modules = ['craype-accel-nvidia60']
# WIP         self.prgenv_flags = {
# WIP             'PrgEnv-cray': ['-std=c++14', '-g', '-O2', '-DNDEBUG',
# WIP                             '-fopenmp'],
# WIP             # Makefile adds: '-DUSE_MPI', '-DUSE_OMP_TARGET'],
# WIP         }
# WIP         # mn CC="cc" CXX="CC"  MPICXX=CC SRCDIR=. BUILDDIR=. BINDIR=.
# WIP         # CUDA_PATH=$CUDATOOLKIT_HOME mpi+omp+target
# WIP         #
# WIP         # CC -DNDEBUG -std=c++14 -O2 -Wall -Wextra -fopenmp -fopenacc
# WIP         # -march=native -mtune=native  -Isrc -Iinclude -DUSE_MPI
# WIP         # -DUSE_OMP_TARGET  src/sqpatch/sqpatch.cpp
# WIP         # -o ./mpi+omp+target.app
# WIP         self.build_system = 'Make'
# WIP         self.build_system.makefile = 'Makefile'
# WIP         self.executable = 'mpi+omp+target'
# WIP         self.build_system.options = [
# WIP             self.executable, 'MPICXX=CC', 'SRCDIR=.', 'BUILDDIR=.',
# WIP             'BINDIR=.', 'CUDA_PATH=$CUDATOOLKIT_HOME',
# WIP             # The makefile adds -DUSE_MPI -DUSE_OMP_TARGET
# WIP             # 'CXXFLAGS=',
# WIP         ]
# WIP         self.sourcesdir = 'src_gpu'  # doublon
# WIP         self.build_system.cxx = 'CC'
# WIP         self.build_system.max_concurrency = 2
# WIP         # }}}
# WIP
# WIP         # {{{ run
# WIP         ompthread = 1
# WIP         self.num_tasks = mpi_task
# WIP         self.num_tasks_per_node = 1
# WIP         self.num_tasks_per_core = 1
# WIP         self.use_multithreading = False
# WIP         self.num_cpus_per_task = ompthread
# WIP         self.exclusive = True
# WIP         self.time_limit = '10m'
# WIP         self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
# WIP         self.cubeside = cubeside_dict[mpi_task]
# WIP         self.steps = steps_dict[mpi_task]
# WIP        self.executable_opts = [f"-n {self.cubeside}", f"-s {self.steps}"]
# WIP         self.prerun_cmds += [
# WIP             'module rm xalt',
# WIP             f'mv {self.executable}.app {self.executable}'
# WIP         ]
# WIP         # }}}
# WIP
# WIP         # {{{ sanity
# WIP         self.sanity_patterns = \
# WIP        sn.assert_found(r'Total time for iteration\(0\)', self.stdout)
# WIP         # }}}
# WIP
# WIP         # {{{ performance
# WIP         # {{{ internal timers
# WIP         # use linux date as timer:
# WIP         self.prerun_cmds += ['echo starttime=`date +%s`']
# WIP         self.postrun_cmds += ['echo stoptime=`date +%s`']
# WIP         # }}}
# WIP
# WIP         # {{{ perf_patterns:
# WIP         self.perf_patterns = sn.evaluate(sphs.basic_perf_patterns(self))
# WIP         # }}}
# WIP
# WIP         # {{{ reference:
# WIP         self.reference = sn.evaluate(sphs.basic_reference_scoped_d(self))
# WIP         # }}}
# WIP         # }}}
# WIP
# WIP     # {{{ hooks
# WIP     @rfm.run_before('compile')
# WIP     def set_compiler_flags(self):
# WIP         self.build_system.cxxflags = \
# WIP             self.prgenv_flags[self.current_environ.name]
# WIP     # }}}
# WIP # }}}
