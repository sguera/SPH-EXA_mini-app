import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
class SphExaMiniAppSquarepatchBaseTest(rfm.RegressionTest):
    """
    https://gperftools.github.io/gperftools/cpuprofile.html
    """
    def __init__(self):
        super().__init__()
        self.descr = 'Strong scaling study'
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-I./include', '-std=c++14', '-g', '-O3',
                           '-fopenmp', '-DUSE_MPI', '-DNDEBUG'],
            'PrgEnv-intel': ['-I./include', '-std=c++14', '-g', '-O3',
                             '-qopenmp', '-DUSE_MPI', '-DNDEBUG'],
            'PrgEnv-cray': ['-I./include', '-hstd=c++14', '-g', '-O3',
                            '-homp', '-DUSE_MPI', '-DNDEBUG'],
            'PrgEnv-pgi': ['-I./include', '-std=c++14', '-g', '-O3',
                           '-mp', '-DUSE_MPI', '-DNDEBUG'],
        }
        # external pprof is needed to avoid "stack trace depth >= 2**32" errors
        self.modules = ['gperftools', 'graphviz', 'pprof']
        self.build_system = 'SingleSource'
        self.testname = 'sqpatch'
        self.sourcepath = '%s.cpp' % self.testname
        self.split_file = 'gperftools.sh'
        # TODO: pass exe + srun flags as args to gperftools.sh
        # self.executable = self.split_file
        self.executable = '%s.exe' % self.testname
        self.executable_opts = ['-s 1']
        self.rpt_file_txt = '%s.txt' % self.executable
        self.rpt_file_pdf = '%s.pdf' % self.executable
        self.rpt_file_doc = '%s.doc' % self.executable
        self.rpt_file_svg = '%s.svg' % self.executable
        # 
        self.maintainers = ['JG']
        self.tags = {'pasc'}
        # self.postbuild_cmd = ['file %s &> %s' % (self.executable, self.rpt)]
        self.sanity_patterns = sn.assert_found('Iteration: 1', self.stdout)

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cxxflags = prgenv_flags
        self.build_system.ldflags = prgenv_flags + ['`pkg-config --libs libprofiler`']

        if self.cpubind:
            self.job.launcher.options = ['--cpu-bind=verbose,%s' % self.cpubind]
        else:
            self.job.launcher.options = ['--cpu-bind=verbose']
        # self.pre_run = [
        #     'echo \'#!/bin/bash\' &> %s' % self.split_file,
        #     'echo \'CPUPROFILE=`hostname`.$SLURM_PROCID\' %s %s >> %s' %
        #     (self.exe, self.exe_opts, self.split_file),
        #    'chmod u+x %s' % (self.split_file),
        # ]
        self.post_run = [
            'srun %s' % self.split_file,
            '$EBROOTPPROF/bin/pprof --unit=ms --text --lines %s %s &> %s' %
            (self.executable, '*.0', self.rpt_file_txt),
            '$EBROOTPPROF/bin/pprof --unit=ms --text --functions %s %s >> %s' %
            (self.executable, '*.0', self.rpt_file_txt),
            '$EBROOTPPROF/bin/pprof --pdf %s %s &> %s' %
            (self.executable, '*.0', self.rpt_file_pdf),
            # 'file %s &> %s' % (self.rpt_file_pdf, self.rpt_file_doc)
            'wget %s' % 'https://raw.githubusercontent.com/brendangregg/'
                         'FlameGraph/master/flamegraph.pl',
            '$EBROOTGPERFTOOLS/bin/pprof --collapsed %s %s > %s.collapsed' %
            (self.executable, '*.0', self.rpt_file_txt),
            'perl flamegraph.pl %s.collapsed > %s.svg' %
            (self.rpt_file_txt, self.rpt_file_svg)
        ]


@rfm.parameterized_test(*[[mpitask, cpubind]
                         for mpitask in [48, 96]
                         # for mpitask in [12, 24, 48, 96]
                         for cpubind in ['none']
                         ])
class SphExaMiniAppSquarepatchHaswellTest(SphExaMiniAppSquarepatchBaseTest):
    # def __init__(self, mpitask, cpubind, prg):
    def __init__(self, mpitask, cpubind):
        super().__init__()
        ompthread = '{:03d}'.format(int(12/mpitask))
        # sysname = prg[0]
        # prgenv = prg[1]
        # compilerversion = prg[2]
        self.cpubind = cpubind
        self.name = 'sqpatch_gperftools' + \
                    '_' + '{:03d}'.format(mpitask) + 'mpi' + \
                    '_' + str(ompthread) + 'omp' + \
                    '-cc' + self.cpubind
        # print(self.name)
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-intel']
        # self.modules = ['Score-P/5.0-CrayIntel-19.03']
        # self.testname = 'sqpatch'
        # self.testname = 'evrard'
        self.time_limit = (0, 15, 0)
        self.exclusive = True
        self.num_tasks = mpitask
        self.num_tasks_per_node = 12
        self.num_cpus_per_task = 1
        # self.num_tasks_per_node = mpitask
        # self.num_cpus_per_task = int(ompthread)
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_PROC_BIND': 'true',
        }

