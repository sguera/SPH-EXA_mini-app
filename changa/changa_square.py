import os
import reframe as rfm
import reframe.utility.sanity as sn

# reframe>=2.13
# module load daint-gpu Python-bare
# reframe --keep-stage-files --prefix=$SCRATCH/sphexa -r -c ./changa_square.py

@rfm.simple_test
class ChangaCpuCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        #super().__init__()
        super().__init__('changa_cpu_check', os.path.dirname(__file__), **kwargs)
        self.descr = ('ReFrame Changa RotSquare3D check')
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-cray']
        #self.sourcesdir = None
        self.sourcesdir = 'src/'
        self.modules = ['ChaNGa/rotsquaretensilecorr-CrayCCE-17.08']
        self.executable = '$EBROOTCHANGA/bin/ChaNGa'
        self.executable_opts = ['+ppn 11 -killat 1 ./run.param']
        self.num_tasks = 16
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12
        # 1 electrical group = 12c/cn*384cn = 4608cores
#         self.extra_resources = {
#             'switches': {
#                 'num_switches': 1
#             }
#         }
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'CRAY_CUDA_MPS': '1',
        }
        self.maintainers = ['JG']
        self.tags = {'sphexa'}

        self.pre_run = [
            'echo SLURM_JOBID=$SLURM_JOBID',
            'echo HUGETLB_MORECORE=$HUGETLB_MORECORE',
            'echo HUGETLB_DEFAULT_PAGE_SIZE=$HUGETLB_DEFAULT_PAGE_SIZE',
            'echo EBROOTCHANGA=$EBROOTCHANGA',
            'module list -t'
        ]

        self.sanity_patterns = sn.assert_eq(sn.count(sn.findall(r'EXIT HYBRID API',
            self.stdout)), self.num_tasks)
