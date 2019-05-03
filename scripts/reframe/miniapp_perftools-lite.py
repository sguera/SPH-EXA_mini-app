import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
class SphExaMiniAppSquarepatchBaseTest(rfm.RegressionTest):
    """
    https://user.cscs.ch/computing/analysis/craypat/
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
        self.modules = ['perftools-lite']
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
        super().setup(partition, environ, **job_opts)
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cxxflags = prgenv_flags
        if self.cpubind:
            self.job.launcher.options = ['--cpu-bind=verbose,%s' % self.cpubind]
        else:
            self.job.launcher.options = ['--cpu-bind=verbose']


@rfm.parameterized_test(*[[mpitask, cpubind]
                         for mpitask in [12, 24, 48, 96]
                         for cpubind in ['none']
                         ])
class SphExaMiniAppSquarepatchHaswellTest(SphExaMiniAppSquarepatchBaseTest):
    def __init__(self, mpitask, cpubind):
        super().__init__()
        ompthread = '{:03d}'.format(int(12/mpitask))
        self.cpubind = cpubind
        self.name = 'sqpatch_perftoolslite' + \
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

