import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher

chicoma_config = Config(
    executors=[
        HighThroughputExecutor(
            label='single_gpu_per_worker',
            cores_per_worker=1,
            available_accelerators=4,
            provider=SlurmProvider(
                partition="gpu",
                account="t24_ml-amd_g",
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 32,
                nodes_per_block=1,
                #scheduler_options='#SBATCH --ntasks-per-node=4 --nodes=1 --ntasks=4 --cpus-per-task=32 --gpus-per-task=1',
                launcher=SimpleLauncher(),
                walltime='10:00:00',
                worker_init="export OMP_NUM_THREADS=32; source ~/.bashrc; conda activate base; source ~/.bash_profile; load_vasp_env",  
            ),
        ),
        HighThroughputExecutor(
            label='cpu_single_node',
            cores_per_worker=128,
            provider=SlurmProvider(
                partition="standard",
                account="t24_ml-amd",
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='00:30:00',
                worker_init="export OMP_NUM_THREADS=128; source ~/.bashrc; conda activate base;",  
            ),
        )
    ],
) 
