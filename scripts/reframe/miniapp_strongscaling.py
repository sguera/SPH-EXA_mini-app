import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
class SphExaMiniAppSquarepatchBaseTest(rfm.RegressionTest):
    """
    cd sph-exa_mini-app.git/scripts/reframe/
    reframe --system dom:mc --exec-policy=async --keep-stage-files \
            --prefix=$SCRATCH/reframe/ -r -c ./miniapp.py
    """
    def __init__(self):
        super().__init__()
        self.descr = 'Strong scaling study'
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-I./include', '-std=c++14', '-O3', '-g',
                           '-fopenmp', '-DUSE_MPI'],
            'PrgEnv-intel': ['-I./include', '-std=c++14', '-O3', '-g',
                             '-qopenmp', '-DUSE_MPI'],
            'PrgEnv-cray': ['-I./include', '-hstd=c++14', '-O3', '-g',
                            '-homp', '-DUSE_MPI'],
            'PrgEnv-pgi': ['-I./include', '-std=c++14', '-O3', '-g',
                           '-mp', '-DUSE_MPI'],
        }
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic'
        }
        self.build_system = 'SingleSource'
        self.testname = 'sqpatch'
        self.sourcepath = '%s.cpp' % self.testname
        self.executable = '%s.exe' % self.testname
        self.rpt = '%s.rpt' % self.testname
        self.maintainers = ['JG']
        self.tags = {'pasc'}
        self.postbuild_cmd = ['file %s &> %s' % (self.executable, self.rpt)]
        # self.sanity_patterns = sn.assert_found(
        #    'ELF 64-bit LSB executable, x86-64', self.rpt)

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cxxflags = prgenv_flags


@rfm.required_version('>=2.14')
@rfm.parameterized_test(*[[mpitask, prg]
                        for mpitask in [1, 2, 3, 4, 6, 9, 12, 18, 36]
                        for prg in [
                            ('dom', 'PrgEnv-gnu', 'gcc/7.3.0'),
                            ('dom', 'PrgEnv-gnu', 'gcc/8.3.0'),
                            ('dom', 'PrgEnv-intel', 'intel/18.0.2.199'),
                            ('dom', 'PrgEnv-intel', 'intel/19.0.1.144'),
                            ('dom', 'PrgEnv-cray', 'cce/8.7.10')]])
class SphExaMiniAppSquarepatchBroadwellTest(SphExaMiniAppSquarepatchBaseTest):
    def __init__(self, mpitask, prg):
        super().__init__()
        ompthread = '{:03d}'.format(int(36/mpitask))
        sysname = prg[0]
        prgenv = prg[1]
        compilerversion = prg[2]
        self.name = 'sphexa_' + compilerversion.replace('/', '') + \
                    '_' + '{:03d}'.format(mpitask) + 'mpi' + \
                    '_' + str(ompthread) + 'omp' + \
                    '_' + sysname
        # print(self.name)
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.valid_prog_environs = [prgenv]
        self.modules = [compilerversion]
        self.time_limit = (0, 10, 0)
        self.exclusive = True
        self.num_tasks = mpitask
        self.num_tasks_per_node = mpitask
        self.num_cpus_per_task = int(ompthread)
        self.num_tasks_per_core = 1
        self.use_multithreading = False


@rfm.required_version('>=2.14')
@rfm.parameterized_test(*[[mpitask, prg]
                        for mpitask in [1, 2, 3]
                        for prg in [
                            ('daint', 'PrgEnv-gnu', 'gcc/4.9.3'),
                            ('daint', 'PrgEnv-gnu', 'gcc/5.3.0'),
                            ('daint', 'PrgEnv-gnu', 'gcc/6.2.0'),
                            ('daint', 'PrgEnv-gnu', 'gcc/7.3.0'),
                            ('daint', 'PrgEnv-intel', 'intel/17.0.4.196'),
                            ('daint', 'PrgEnv-intel', 'intel/18.0.2.199'),
                            ('daint', 'PrgEnv-cray', 'cce/8.6.1'),
                            ('daint', 'PrgEnv-cray', 'cce/8.7.4'),
                            ('daint', 'PrgEnv-pgi', 'pgi/17.5.0'),
                            ('daint', 'PrgEnv-pgi', 'pgi/18.5.0'),
                            ('daint', 'PrgEnv-pgi', 'pgi/18.10.0')]])
class SphExaMiniAppSquarepatchHaswellTest(SphExaMiniAppSquarepatchBaseTest):
    def __init__(self, mpitask, prg):
        super().__init__()
        ompthread = '{:03d}'.format(int(12/mpitask))
        sysname = prg[0]
        prgenv = prg[1]
        compilerversion = prg[2]
        self.name = 'sphexa_' + compilerversion.replace('/', '') + \
                    '_' + '{:03d}'.format(mpitask) + 'mpi' + \
                    '_' + str(ompthread) + 'omp' + \
                    '_' + sysname
        # print(self.name)
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = [prgenv]
        self.modules = [compilerversion]
        self.time_limit = (0, 10, 0)
        self.exclusive = True
        self.num_tasks = mpitask
        self.num_tasks_per_node = mpitask
        self.num_cpus_per_task = int(ompthread)
        self.num_tasks_per_core = 1
        self.use_multithreading = False
