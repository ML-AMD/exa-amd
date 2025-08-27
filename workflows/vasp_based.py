import os
import time
import pandas as pd
import glob

from tools.errors import VaspNonReached
from parsl.app.errors import AppTimeout
from parsl.app.errors import BashExitFailure
from tools.logging_config import amd_logger
from tools.config_manager import ConfigManager
from tools.config_labels import ConfigKeys as CK


from tools.post_processing import get_vasp_hull

STATUS_BY_EXCEPTION = {
    VaspNonReached:  "non_reached",
    AppTimeout:      "time_out",
    BashExitFailure: "bash_exit_failure",
}
def write_status(fp, id_, status):
    fp.write(f"{id_},{status}\n")

def vasp_calculations(config):
    from parsl_tasks.dft_optimization import run_vasp_calc
    work_dir = config[CK.WORK_DIR]
    output_file_vasp_calc = os.path.join(
        config[CK.VASP_WORK_DIR], config[CK.OUTPUT_FILE])

    # open the output file to log the structures that failed or succeded to
    # converge
    fp = open(output_file_vasp_calc, 'w')
    fp.write("id,result\n")

    # launch all vasp calculations
    l_futures = [(run_vasp_calc(config.get_json_config(), i), i)
                 for i in range(config["nstart"], config["nend"])]

    # wait for all the tasks (in the batch) to complete
    for future, id in l_futures:
        try:
            err = future.exception()
            if err:
                raise err
            write_status(fp, id, "success")
        except tuple(STATUS_BY_EXCEPTION) as e:
            write_status(fp, id, STATUS_BY_EXCEPTION[type(e)])
        except Exception as e:
            amd_logger.warning(f"An exception occurred: {e}")
            write_status(fp, id, "unexpected_error")

    fp.close()


def generate_structures(config):
    from parsl_tasks.gen_structures import gen_structures
    try:
        n_chunks = config[CK.GEN_STRUCTURES_NNODES]
        l_futures = [gen_structures(config.get_json_config(), n_chunks, i) for i in range(1, n_chunks + 1)]

        for future in l_futures:
            future.result()

    except Exception as e:
        amd_logger.critical(f"An exception occurred: {e}")


def select_structures(config):
    from parsl_tasks.select_structures import select_structures
    try:
        select_structures(config.get_json_config()).result()
    except Exception as e:
        amd_logger.critical(f"An exception occurred: {e}")


def run_cgcnn(config):
    from parsl_tasks.cgcnn import cgcnn_prediction
    try:
        n_chunks = config[CK.GEN_STRUCTURES_NNODES]
        l_futures = [cgcnn_prediction(config.get_json_config(), n_chunks, i) for i in range(1, n_chunks + 1)]

        for future in l_futures:
            future.result()

        # merge results
        pattern = os.path.join(config[CK.WORK_DIR], "test_results_*.csv")
        files_to_merge = list(glob.iglob(pattern))
        if not files_to_merge:
            return

        dataframes = pd.concat(
            (pd.read_csv(file, header=None) for file in files_to_merge),
            ignore_index=True
        )
        dataframes.to_csv(os.path.join(config[CK.WORK_DIR], "test_results.csv"), index=False)

        # cleanup
        for file in files_to_merge:
            os.remove(file)

    except Exception as e:
        amd_logger.critical(f"An exception occurred: {e}")


def post_processing(config):
    from parsl_tasks.hull import calculate_ehul
    from parsl_tasks.hull import convex_hull_color

    if config[CK.POST_PROCESSING_OUT_DIR]:

        elements = config[CK.ELEMENTS]
        nb_of_elements = len(elements.split('-'))

        if nb_of_elements < 3 or nb_of_elements > 4:
            amd_logger.critical(
                f"The post-processing is only supported with 3 or 4 elements")

        os.makedirs(config[CK.POST_PROCESSING_OUT_DIR], exist_ok=True)

        get_vasp_hull(config)

        err = calculate_ehul(config).exception()
        if err:
            amd_logger.critical(err)

        err = convex_hull_color(config).exception()
        if err:
            amd_logger.critical(err)

        out_hull = os.path.join(
            config[CK.POST_PROCESSING_OUT_DIR], CK.POST_PROCESSING_FINAL_OUT)
        amd_logger.info(f"Convex hull plot saved to '{out_hull}'")

        out_selected = os.path.join(
            config[CK.POST_PROCESSING_OUT_DIR], "selected")
        amd_logger.info(f"Selected candidates saved to '{out_selected}'")


def run_workflow(config):
    """
    Run the full VASP-based materials discovery workflow.

    Consists of the following task-based steps (with a dependency between each step):

    1. **Structure Generation**
       :func:`~parsl_tasks.gen_structures.generate_structures`

    2. **CGCNN Prediction**
       :func:`~parsl_tasks.cgcnn.run_cgcnn`.

    3. **Structure Selection**
       :func:`~parsl_tasks.cgcnn.select_structures`.

    4. **VASP Calculations**
       :func:`~parsl_tasks.vasp.vasp_calculations`

    5. **Post Processing**

    Args:
        config (ConfigManager): The configuration manager that provides runtime parameters,
            paths, and thresholds for each stage of the workflow.

    Side Effects:
        - Creates directories and files under `config[CK.WORK_DIR]`
        - Executes multiple shell commands and external applications

    Raises:
        Exception: If any sub-stage raises an error that is not internally handled.
    """
    amd_logger.info("Start the 'vasp_based' workflow'")

    if not os.path.exists(os.path.join(
            config[CK.WORK_DIR], 'structures/1')):
        generate_structures(config)
    amd_logger.info(f"generate_structures done")

    if not os.path.exists(os.path.join(
            config[CK.WORK_DIR], 'test_results.csv')):
        run_cgcnn(config)
    amd_logger.info(f"cgcnn done")

    if not os.path.exists(os.path.join(config[CK.WORK_DIR], 'new/POSCAR_1')):
        select_structures(config)
    amd_logger.info(f"select structures done")

    config.setup_vasp_calculations()
    vasp_calculations(config)
    amd_logger.info(f"vasp calculations done")

    post_processing(config)
    amd_logger.info(f"post_processing done")
