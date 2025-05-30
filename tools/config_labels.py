"""
Centralized mapping of internal config keys to user-visible CLI labels.
"""


class ConfigKeys:
    # required
    CMS_DIR = "cms_dir"
    VASP_STD_EXE = "vasp_std_exe"
    WORK_DIR = "work_dir"
    VASP_WORK_DIR = "vasp_work_dir"
    POT_DIR = "vasp_pot_dir"
    OUTPUT_FILE = "vasp_output_file"
    ELEMENTS = "elements"
    PARSL_CONFIG = "parsl_config"
    CPU_ACCOUNT = "cpu_account"
    GPU_ACCOUNT = "gpu_account"
    CONFIG_FILE = "config"
    INITIAL_STRS = "initial_structures_dir"

    # optional
    EF_THR = "formation_energy_threshold"
    NUM_WORKERS = "num_workers"
    BATCH_SIZE = "cgcnn_batch_size"
    VASP_NNODES = "vasp_nnodes"
    VASP_NTASKS_PER_RUN = "vasp_ntasks_per_run"
    NUM_STRS = "vasp_nstructures"
    VASP_TIMEOUT = "vasp_timeout"
    VASP_NSW = "vasp_nsw"
    OUTPUT_LEVEL = "output_level"
