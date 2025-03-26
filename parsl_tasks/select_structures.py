from parsl import bash_app
from parsl_configs.parsl_executors_labels import CPU_SINGLE_LABEL


@bash_app(executors=[CPU_SINGLE_LABEL])
def select_structures(config):
    import os
    try:
        os.chdir(config["work_dir"])

        tr_csv_file = os.path.join(config["work_dir"], "test_results.csv")
        dir_structures = os.path.join(config["work_dir"], "structures")
        dir_select_structure = os.path.join(
            config["cms_dir"], "select_structure.py")

    except Exception as e:
        raise
    return "python {} --ef_threshold {} --num_workers {} --csv_file {} --nomix_dir {}".format(dir_select_structure, str(config["ef_thr"]), config["num_workers"], tr_csv_file, dir_structures)

@bash_app(executors=[CPU_SINGLE_LABEL])
def select_structures_init_perf():
    return "ls"
