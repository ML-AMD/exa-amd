import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher

exec_config = Config(
    executors=[
        HighThroughputExecutor(
            label='single_gpu_per_worker',
            cores_per_worker=32,
            available_accelerators=4,
            provider=SlurmProvider(
                #partition="gpu",
                account="m4802_g",
                qos="regular",
                constraint="gpu",
                #scheduler_options='#SBATCH --ntasks-per-node=4',
                init_blocks = 1,
                min_blocks = 1,
                max_blocks = 1,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='00:20:00',
                worker_init="export OMP_NUM_THREADS=32; module load python/3.11; module load vasp/6.4.3-gpu",  
            ),
        )
    ],
) 
