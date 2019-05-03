import reframe as rfm
import reframe.utility.sanity as sn


class SphExaMiniAppSquarepatchBaseTest(rfm.RegressionTest):
    """
    http://llnl.github.io/mpiP
    """
    def __init__(self):
        super().__init__()
        self.descr = 'Strong scaling study'
        # -g compilation flag is needed to report source code filename and line
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
        mpip_ver = '57fc864'
        tc_ver = '19.03'
        self.mpip_modules = {
            'PrgEnv-gnu': ['mpiP/%s-CrayPGI-%s' % (mpip_ver, tc_ver)],
            'PrgEnv-intel': ['mpiP/%s-CrayIntel-%s' % (mpip_ver, tc_ver)],
            'PrgEnv-cray': ['mpiP/%s-CrayCCE-%s' % (mpip_ver, tc_ver)],
            'PrgEnv-pgi': ['mpiP/%s-CrayPGI-%s' % (mpip_ver, tc_ver)],
        }
        self.build_system = 'SingleSource'
        self.testname = 'sqpatch'
        self.sourcepath = '%s.cpp' % self.testname
        self.executable = '%s.exe' % self.testname
        self.executable_opts = ['-s 25']
        self.rpt_file_txt = '%s.txt' % self.executable
        self.maintainers = ['JG']
        self.tags = {'pasc'}
        self.sanity_patterns = sn.assert_found('Iteration: 1', self.stdout)

    def setup(self, partition, environ, **job_opts):
        self.modules = self.mpip_modules[environ.name]
        super().setup(partition, environ, **job_opts)
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cxxflags = prgenv_flags
        if self.cpubind:
            self.job.launcher.options = ['--cpu-bind=verbose,%s' % self.cpubind]
        else:
            self.job.launcher.options = ['--cpu-bind=verbose']
        self.build_system.ldflags = prgenv_flags + ['-L$(EBROOTMPIP)/lib',
                                             '-Wl,--whole-archive -lmpiP',
                                             '-Wl,--no-whole-archive -lunwind',
                                             '-lbfd -liberty -ldl -lz']


@rfm.parameterized_test(*[[mpitask, cpubind]
                         for mpitask in [12, 24, 48, 96]
                         for cpubind in ['none']
                         ])
class SphExaMiniAppSquarepatchHaswellTest(SphExaMiniAppSquarepatchBaseTest):
    def __init__(self, mpitask, cpubind):
        super().__init__()
        ompthread = '{:03d}'.format(int(12/mpitask))
        self.cpubind = cpubind
        self.name = 'sqpatch_mpip' + \
                    '_' + '{:03d}'.format(mpitask) + 'mpi' + \
                    '_' + str(ompthread) + 'omp' + \
                    '-cc' + self.cpubind
        # print(self.name)
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.time_limit = (2, 0, 0)
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

