import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher
from parsl.launchers import SrunLauncher

from parsl_configs.parsl_config_registry import register_parsl_config
from parsl_configs.parsl_executors_labels import SINGLE_GPU_LABEL, CPU_SINGLE_LABEL

class PerlmutterPremiumConfig(Config):
    def __init__(self, json_config):
        """
          - json_config["vasp_nnodes"] (int): number of GPU nodes used for VASP calculations
          - json_config["num_workers"] (int): number of CPU workers per node
        """

        nnodes_vasp = json_config["vasp_nnodes"]
        num_workers = json_config["num_workers"]

        # GPU executor
        single_gpu_per_worker_executor = HighThroughputExecutor(
            label=SINGLE_GPU_LABEL,
            cores_per_worker=1,
            available_accelerators=4,
            provider=SlurmProvider(
                account="m4802_g",
                qos="premium",
                constraint="gpu",
                init_blocks=0,
                min_blocks=nnodes_vasp,
                max_blocks=nnodes_vasp,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='16:00:00',
                worker_init="conda activate amd_env; module load vasp/6.4.3-gpu",
            )
        )

        # CPU executor
        cpu_single_node_executor = HighThroughputExecutor(
            label=CPU_SINGLE_LABEL,
            cores_per_worker=128,
            provider=SlurmProvider(
                account="m4802",
                qos="premium",
                constraint="cpu",
                init_blocks=0,
                min_blocks=1,
                max_blocks=1,
                nodes_per_block=1,
                launcher=SimpleLauncher(),
                walltime='01:00:00',
                worker_init="conda activate amd_env;"
            )
        )

        super().__init__(
            executors=[single_gpu_per_worker_executor, cpu_single_node_executor])

# Register the perlmutter configs
register_parsl_config("perlmutter_premium", PerlmutterPremiumConfig)
