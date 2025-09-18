import parsl
import math
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher
from parsl.launchers import SrunLauncher
from parsl.launchers import SingleNodeLauncher

from parsl_configs.parsl_config_registry import register_parsl_config
from parsl_configs.parsl_executors_labels import *
from tools.config_labels import ConfigKeys as CK


class PerlmutterConfig(Config):
    """
    Parsl configuration for running exa-AMD on Perlmutter.

    This configuration defines four Parsl executors, each targeting a different part of the
    exa-AMD workflow and resource type:

    - **Generate Structures Executor** (`generate_structures_executor`): run on CPU, multi-node.
    - **CGCNN Executor** (`cgcnn_executor`): run on GPU, multi-node.
    - **Select Structures Executor** (`select_structures_executor`): run on CPU, single-node.
    - **VASP Executor** (`vasp_executor`): run on GPU, multi-node.
    - **Post-processing Executor** (`post_processing`): run on GPU, multi-node.

    Args:
        json_config (dict)
            Required keys:
                - ``vasp_nnodes`` (int): Number of GPU nodes to use for VASP calculations.
                - ``num_cores`` (int): Number of workers to allocate per CPU node.
    """

    def __init__(self, json_config):
        """
          - json_config[CK.GEN_STRUCTURES_NNODES] (int): number of CPU (and GPU) nodes used for generating the structures (and formation energy prediction)
          - json_config[CK.VASP_NNODES] (int): number of GPU nodes used for VASP calculations
          - json_config[CK.NUM_WORKERS] (int): number of CPU workers per node
          - json_config[CK.CPU_ACCOUNT] (int): slurm CPU account
          - json_config[CK.GPU_ACCOUNT] (int): slurm GPU account
        """

        nnodes_vasp = json_config[CK.VASP_NNODES]
        nnodes_gen_struct = json_config[CK.GEN_STRUCTURES_NNODES]
        nnodes_cgcnn = nnodes_gen_struct
        num_cores = json_config[CK.NUM_WORKERS]
        num_cores_cgcnn = num_cores
        cpu_account = json_config[CK.CPU_ACCOUNT]
        gpu_account = json_config[CK.GPU_ACCOUNT]

        print("# nnodes_cgcnn ", nnodes_cgcnn)

        # VASP executor
        vasp_executor = HighThroughputExecutor(
            label=VASP_EXECUTOR_LABEL,
            cores_per_worker=1,
            available_accelerators=4,
            provider=SlurmProvider(
                account=gpu_account,
                qos="premium",
                constraint="gpu",
                init_blocks=0,
                min_blocks=nnodes_vasp,
                max_blocks=nnodes_vasp,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='16:00:00',
                worker_init="module load vasp/6.4.3-gpu",
            )
        )

        cgcnn_executor = HighThroughputExecutor(
            label=CGCNN_EXECUTOR_LABEL,
            max_workers_per_node=1,
            available_accelerators=1,
            cores_per_worker=num_cores_cgcnn,
            provider=SlurmProvider(
                account=gpu_account,
                qos="premium",
                constraint="gpu",
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=nnodes_cgcnn,
                launcher=SingleNodeLauncher(),
                walltime="01:00:00",
                worker_init="module load conda/Miniforge3-24.7.1-0 && conda activate amd_env",
                scheduler_options="#SBATCH --cpus-per-task=128\n#SBATCH --exclusive"
            )
        )

        # generate executor
        generate_structures_executor = HighThroughputExecutor(
            label=GENERATE_EXECUTOR_LABEL,
            cores_per_worker=num_cores,
            max_workers_per_node=1,
            provider=SlurmProvider(
                account=cpu_account,
                qos="premium",
                constraint="cpu",
                init_blocks=0,
                min_blocks=1,
                max_blocks=1,
                nodes_per_block=nnodes_gen_struct,
                launcher=SrunLauncher(),
                walltime='01:00:00',
                worker_init="module load conda/Miniforge3-24.7.1-0 && conda activate amd_env",
                scheduler_options="#SBATCH --exclusive",
            )
        )

        # select executor
        select_structures_executor = HighThroughputExecutor(
            label=SELECT_EXECUTOR_LABEL,
            cores_per_worker=num_cores,
            max_workers_per_node=1,
            provider=SlurmProvider(
                account=cpu_account,
                qos="premium",
                constraint="cpu",
                init_blocks=0,
                min_blocks=1,
                max_blocks=1,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='01:00:00',
                worker_init="module load conda/Miniforge3-24.7.1-0 && conda activate amd_env",
            )
        )

        # Post-processing executor
        post_processing_executor = HighThroughputExecutor(
            label=POSTPROCESSING_LABEL,
            cores_per_worker=1,
            available_accelerators=4,
            provider=SlurmProvider(
                account=gpu_account,
                qos="premium",
                constraint="gpu",
                init_blocks=0,
                min_blocks=1,
                max_blocks=1,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='5:00:00',
                worker_init="module load vasp/6.4.3-gpu",
            )
        )

        super().__init__(
            executors=[vasp_executor, cgcnn_executor, generate_structures_executor, select_structures_executor, post_processing_executor])


# Register the perlmutter configs
register_parsl_config("perlmutter_premium", PerlmutterConfig)
