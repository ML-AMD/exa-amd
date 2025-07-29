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
    POST_PROCESSING_OUT_DIR = "post_processing_output_dir"
    MPRester_API_KEY = "mp_rester_api_key"
    HULL_ENERGY_THR = "hull_energy_threshold"
    GEN_STRUCTURES_NNODES = "gen_structure_nnodes"

    # hardcoded keys
    SUBDIR_STABLE_PHASES = "stable_phases_work_dir"
    MP_STABLE_OUT = "mp_int_stable.dat"
    ENERGY_DAT_OUT = "energy.dat"
    POST_PROCESSING_FINAL_OUT = "hull_plot.png"
