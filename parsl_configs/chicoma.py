import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher
from parsl.launchers import SrunLauncher
from parsl.launchers import SingleNodeLauncher
from parsl.data_provider.file_noop import NoOpFileStaging

chicoma_config = Config(
    executors=[
        HighThroughputExecutor(
            label='chicoma_single_gpu_per_worker',
            cores_per_worker=32,
            available_accelerators=4,
            provider=SlurmProvider(
                partition="gpu",
                account="t24_ml-amd_g",
                init_blocks = 1,
                min_blocks = 1,
                max_blocks = 1,
                nodes_per_block=1,
                #scheduler_options='#SBATCH --ntasks-per-node=4 --nodes=1 --ntasks=4 --cpus-per-task=32 --gpus-per-task=1',
                launcher=SimpleLauncher(),
                walltime='10:00:00',
                worker_init="export OMP_NUM_THREADS=32; source ~/.bashrc; conda activate base; source ~/.bash_profile; load_vasp_env",  
                #scheduler_options='#SBATCH --nodes=1 -p gpu -A w23_ml4chem_g --qos=debug --reservation=gpu_debug',
            ),
        )
    ],
) 
