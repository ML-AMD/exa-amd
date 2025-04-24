import parsl
import json

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher
from parsl.launchers import SingleNodeLauncher
from parsl.launchers import SrunLauncher
from parsl.providers import LocalProvider
from parsl.launchers import SrunMPILauncher

from parsl_configs.parsl_config_registry import register_parsl_config
from parsl_configs.parsl_executors_labels import *

# environment to set before running a VASP calculation
vasp_env_init = '''
                conda activate amd_env
                export NVHPC_CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5
                export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5
                export CRAY_ACCEL_TARGET=nvidia80

                export MPICH_GPU_SUPPORT_ENABLED=1
                export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

                export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
                export LD_LIBRARY_PATH=$NVHPC_CUDA_HOME/lib64:$LD_LIBRARY_PATH
                module load cray-fftw
                module load cray-hdf5
                ulimit -c 0
            '''


class ChicomaConfig(Config):
    def __init__(self, json_config):
        """
          - json_config["vasp_nnodes"] (int): number of GPU nodes used for VASP calculations
          - json_config["num_workers"] (int): number of CPU workers per node
        """

        nnodes_vasp = json_config["vasp_nnodes"]
        num_workers = json_config["num_workers"]

        # GPU executor
        single_gpu_per_worker_executor = HighThroughputExecutor(
            label=GPU_VASP_EXECUTOR_LABEL,
            cores_per_worker=1,
            available_accelerators=4,
            provider=SlurmProvider(
                partition="gpu",
                account="t25_ml-amd_g",
                init_blocks=0,
                min_blocks=nnodes_vasp,
                max_blocks=nnodes_vasp,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='16:00:00',
                # worker_init=(
                #     "conda activate amd_env; "
                #     "source ~/.bash_profile; load_vasp_env"
                # )
                worker_init=vasp_env_init
            )
        )

        # CPU executor
        cpu_single_node_executor = HighThroughputExecutor(
            label=CPU_GENERATE_EXECUTOR_LABEL,
            cores_per_worker=128,
            provider=SlurmProvider(
                partition="standard",
                account="t25_ml-amd",
                init_blocks=0,
                min_blocks=1,
                max_blocks=1,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='01:00:00',
                worker_init="source ~/.bashrc; conda activate amd_env; export OMP_NUM_THREADS=128"
            )
        )

        super().__init__(
            executors=[single_gpu_per_worker_executor, cpu_single_node_executor])


class ChicomaConfigDebug(Config):
    def __init__(self, json_config):
        """
          - json_config["vasp_nnodes"] (int): number of GPU nodes used for VASP calculations
          - json_config["num_workers"] (int): number of CPU workers per node
        """

        nnodes_vasp = json_config["vasp_nnodes"]
        num_workers = json_config["num_workers"]

        # GPU executor
        single_gpu_per_worker_executor = HighThroughputExecutor(
            label=GPU_VASP_EXECUTOR_LABEL,
            cores_per_worker=1,
            available_accelerators=4,
            provider=SlurmProvider(
                partition="gpu",
                init_blocks=0,
                min_blocks=nnodes_vasp,
                max_blocks=nnodes_vasp,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='01:00:00',
                worker_init=vasp_env_init,
                scheduler_options="#SBATCH --reservation=gpu_debug",
            )
        )

        # CPU executor
        cpu_single_node_executor = HighThroughputExecutor(
            label=CPU_GENERATE_EXECUTOR_LABEL,
            cores_per_worker=128,
            provider=SlurmProvider(
                partition="cpu",
                init_blocks=0,
                min_blocks=1,
                max_blocks=1,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='01:00:00',
                worker_init="conda activate amd_env",
                scheduler_options="#SBATCH --reservation=debug",
            )
        )

        super().__init__(
            executors=[single_gpu_per_worker_executor, cpu_single_node_executor])


# Register the chicoma configs
register_parsl_config("chicoma", ChicomaConfig)
register_parsl_config("chicoma_debug", ChicomaConfigDebug)
