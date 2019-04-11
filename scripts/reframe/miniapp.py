import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(
    ['dom', 'PrgEnv-gnu', 'gcc/7.3.0'],
    ['dom', 'PrgEnv-gnu', 'gcc/8.3.0'],
    ['dom', 'PrgEnv-intel', 'intel/18.0.2.199'],
    ['dom', 'PrgEnv-intel', 'intel/19.0.1.144'],
    ['dom', 'PrgEnv-cray', 'cce/8.7.10'],
    # 
    ['daint', 'PrgEnv-gnu', 'gcc/4.9.3'],
    ['daint', 'PrgEnv-gnu', 'gcc/5.3.0'],
    ['daint', 'PrgEnv-gnu', 'gcc/6.2.0'],
    ['daint', 'PrgEnv-gnu', 'gcc/7.3.0'],
    ['daint', 'PrgEnv-intel', 'intel/17.0.4.196'],
    ['daint', 'PrgEnv-intel', 'intel/18.0.2.199'],
    ['daint', 'PrgEnv-cray', 'cce/8.6.1'],
    ['daint', 'PrgEnv-cray', 'cce/8.7.4'],
    ['daint', 'PrgEnv-pgi', 'pgi/17.5.0'],
    ['daint', 'PrgEnv-pgi', 'pgi/18.5.0'],
    ['daint', 'PrgEnv-pgi', 'pgi/18.10.0'],
)
class SphExaMiniAppSquarepatch(rfm.CompileOnlyRegressionTest):
    """
    cd sph-exa_mini-app.git/scripts/reframe/
    reframe --system dom:mc --exec-policy=async --keep-stage-files \
            --prefix=$SCRATCH/reframe/ -r -c ./miniapp.py
    """
    def __init__(self, sysname, prgenv, compilerversion):
        super().__init__()
        self.name = 'sphexa_' + sysname + "_" + compilerversion.replace('/', '')
        self.descr = 'compilation only check'
        self.valid_systems = ['%s:gpu' % sysname, '%s:mc' % sysname]
        self.valid_prog_environs = [prgenv]
        self.modules = [compilerversion]
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-I./include', '-std=c++14', '-O3', '-g',
                           '-fopenmp', '-D_JENKINS'],
            'PrgEnv-intel': ['-I./include', '-std=c++14', '-O3', '-g',
                             '-qopenmp', '-D_JENKINS'],
            'PrgEnv-cray': ['-I./include', '-hstd=c++14', '-O3', '-g',
                            '-homp', '-D_JENKINS'],
            'PrgEnv-pgi': ['-I./include', '-std=c++14', '-O3', '-g',
                           '-mp', '-D_JENKINS'],
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
        self.sanity_patterns = sn.assert_found(
            'ELF 64-bit LSB executable, x86-64', self.rpt)

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cxxflags = prgenv_flags
