from parsl import bash_app

from parsl_configs.parsl_executors_labels import SELECT_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK


def cmd_select_structures(config):
    """
    Construct the shell command used to run the structure selection script via Parsl.

    Prepares the command to execute the `cms_dir/select_structure.py` script.
    This task is used to identify and remove duplicate or near-duplicate structures,
    based on a structural similarity threshold

    Args:
        config (dict): ConfigManager. The following fields are used:
            - work_dir
            - cms_dir
            - ef_thr
            - num_workers

            See :class:`~tools.config_manager.ConfigManager` for field descriptions.

    Returns:
        str: A formatted shell command string to execute `select_structure.py`.

    Raises:
        Exception: If directory navigation or path generation fails.
    """
    import os
    try:
        os.chdir(config[CK.WORK_DIR])

        tr_csv_file = os.path.join(config[CK.WORK_DIR], "test_results.csv")
        dir_structures = os.path.join(config[CK.WORK_DIR], "structures")
        dir_select_structure = os.path.join(
            config[CK.CMS_DIR], "select_structure.py")

    except Exception as e:
        raise
    return (
        f"python {dir_select_structure} --ef_threshold {config[CK.EF_THR]} "
        f"--num_workers {config[CK.NUM_WORKERS]} --csv_file {tr_csv_file} "
        f"--nomix_dir {dir_structures}"
    )

@bash_app(executors=[SELECT_EXECUTOR_LABEL])
def select_structures(config):
    return cmd_select_structures(config)
