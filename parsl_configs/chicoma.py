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
                partition="gpu_debug",
                account="t24_ml-amd_g",
                qos="debug",
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 32,
                nodes_per_block=1,
                #scheduler_options='#SBATCH --ntasks-per-node=4 --nodes=1 --ntasks=4 --cpus-per-task=32 --gpus-per-task=1',
                launcher=SimpleLauncher(),
                walltime='01:00:00',
                worker_init="export OMP_NUM_THREADS=32; source ~/.bashrc; conda activate base; source ~/.bash_profile; load_vasp_env",  
                scheduler_options='#SBATCH --reservation=gpu_debug'
            ),
        ),
        HighThroughputExecutor(
            label='single_gpu_per_worker_cgcnn',
            cores_per_worker=128,
            available_accelerators=4,
            provider=SlurmProvider(
                partition="gpu_debug",
                account="t24_ml-amd_g",
                qos="debug",
                init_blocks = 0,
                min_blocks = 0,
                max_blocks = 1,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='01:00:00',
                worker_init="export OMP_NUM_THREADS=128; source ~/.bashrc; conda activate base; ",  
                scheduler_options='#SBATCH --reservation=gpu_debug'
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
        ),
    ],
) 

# salloc -p gpu_debug -N 1 -A t24_ml-amd_g -t 1:00:00 --reservation=gpu_debug