import os
import time

from tools.errors import VaspNonReached
from parsl.app.errors import AppTimeout
from parsl.app.errors import BashExitFailure
from tools.logging_config import amd_logger
from tools.config_manager import ConfigManager


def vasp_calculations(config):
    from parsl_tasks.dft_optimization import run_vasp_calc
    work_dir = config["work_dir"]
    output_file_vasp_calc = os.path.join(
        config["vasp_work_dir"], config["output_file"])

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
            fp.write("{},{}\n".format(id, "success"))
        except VaspNonReached:
            fp.write("{},{}\n".format(id, "non_reached"))
        except AppTimeout:
            fp.write("{},{}\n".format(id, "time_out"))
        except BashExitFailure:
            fp.write("{},{}\n".format(id, "bash_exit_failure"))
        except Exception as e:
            amd_logger.warning(f"An exception occurred: {e}")
            fp.write("{},{}\n".format(id, "unexpected_error"))

    fp.close()


def generate_structures(config):
    from parsl_tasks.gen_structures import gen_structures
    try:
        gen_structures(config.get_json_config()).result()
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
        cgcnn_prediction(config.get_json_config()).result()
    except Exception as e:
        amd_logger.critical(f"An exception occurred: {e}")


def run_workflow(config):
    amd_logger.info("Start the 'vasp_based' workflow'")

    if not os.path.exists(os.path.join(
            config["work_dir"], 'structures/1.cif')):
        generate_structures(config)
    amd_logger.info(f"generate_structures done")

    if not os.path.exists(os.path.join(
            config["work_dir"], 'test_results.csv')):
        run_cgcnn(config)
    amd_logger.info(f"cgcnn done")

    if not os.path.exists(os.path.join(config["work_dir"], 'new/POSCAR_1')):
        select_structures(config)
    amd_logger.info(f"select structures done")

    config.setup_vasp_calculations()
    vasp_calculations(config)
    amd_logger.info(f"vasp calculations done")
