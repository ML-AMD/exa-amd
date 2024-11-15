import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher

exec_config_debug = Config(
    executors=[
        HighThroughputExecutor(
            label='single_gpu_per_worker',
            cores_per_worker=32,
            available_accelerators=4,
            provider=SlurmProvider(
                #partition="gpu",
                account="m4802_g",
                qos="debug",
                constraint="gpu",
                #scheduler_options='#SBATCH --ntasks-per-node=4',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 32,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='00:30:00',
                worker_init="export OMP_NUM_THREADS=32; module load python; module load pytorch",  
            ),
        ),
        HighThroughputExecutor(
            label='cpu_single_node',
            cores_per_worker=256,
            provider=SlurmProvider(
                #partition="gpu",
                account="m4802",
                qos="debug",
                constraint="cpu",
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='00:30:00',
                worker_init="export OMP_NUM_THREADS=256; module load python; module load pytorch",  
            ),
        )
    ],
) 

