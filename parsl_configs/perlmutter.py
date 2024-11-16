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
                qos="premium",
                constraint="gpu",
                #scheduler_options='#SBATCH --ntasks-per-node=4',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 5,
                nodes_per_block=2,
                launcher=SimpleLauncher(),
                walltime='24:00:00',
                worker_init="export OMP_NUM_THREADS=1; export OMP_PLACES=threads; export OMP_PROC_BIND=spread; module load vasp/6.4.3-gpu"
            ),
        ),
        HighThroughputExecutor(
            label='single_gpu_per_worker_cgcnn',
            cores_per_worker=128,
            available_accelerators=4,
            provider=SlurmProvider(
                #partition="gpu",
                account="m4802_g",
                qos="premium",
                constraint="gpu",
                #scheduler_options='#SBATCH --ntasks-per-node=4',
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 32,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='00:30:00',
                worker_init="module load pytorch",  
            ),
        ),
        HighThroughputExecutor(
            label='cpu_single_node',
            cores_per_worker=128,
            provider=SlurmProvider(
                #partition="gpu",
                account="m4802_g",
                qos="premium",
                constraint="gpu",
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='00:30:00',
                worker_init="module load pytorch",  
            ),
        )
    ],
) 

