import os
import reframe as rfm
import reframe.utility.sanity as sn

# reframe>=2.13
# reframe --exec-policy async --keep-stage-files --prefix=$SCRATCH/sphexa -r -c ./sphynx_square.py

@rfm.simple_test
class SphynxCpuCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        #super().__init__()
        super().__init__('sphynx_cpu_check', os.path.dirname(__file__), **kwargs)
        self.descr = ('ReFrame Sphynx RotSquare3D check')
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-intel']
        #self.sourcesdir = None
        self.sourcesdir = 'src/'

        #self.modules = ['CrayIntel-17.08']
        self.modules = ['sphynx/1.4-CrayIntel-17.08-squarepatch10M-96-cn']
		# sphynx/1.4-CrayIntel-17.08-square3D10Mp-1-cn
		# sphynx/1.4-CrayIntel-17.08-square3D10Mp-2-cn
		# sphynx/1.4-CrayIntel-17.08-square3D10Mp-8-cn
		# sphynx/1.4-CrayIntel-17.08-square3D10Mp-16-cn
		# sphynx/1.4-CrayIntel-17.08-square3D10Mp-32-cn
		# sphynx/1.4-CrayIntel-17.08-square3D10Mp-64-cn
		# sphynx/1.4-CrayIntel-17.08-square3D10Mp-80-cn
		# sphynx/1.4-CrayIntel-17.08-square3D10Mp-96-cn
        self.executable = '$EBROOTSPHYNX/bin/*.exe'
        self.num_tasks = 96
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12
        # 96cn = 1152c
        # 1 electrical group = 12c/cn*384cn = 4608cores
#         self.extra_resources = {
#             'switches': {
#                 'num_switches': 1
#             }
#         }
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'MPICH_NO_BUFFER_ALIAS_CHECK': '1'
        }
        self.maintainers = ['JG']
        self.tags = {'sphexa'}

        self.pre_run = [
            'echo EBROOTSPHYNX=$EBROOTSPHYNX',
            'module list -t'
        ]

        self.outputtimes_file = 'conservelaws.d'
        self.sanity_patterns = sn.all([
            sn.assert_found('2.1000000000E-04', self.outputtimes_file)
        ])

#        self.sanity_patterns = sn.assert_bounded(sn.extractsingle(
#            r'Random: (?P<number>\S+)', self.stdout, 'number', float),
#            lower, upper)

#def _get_checks(**kwargs):
#    return [SphynxCpuCheck(**kwargs)]
