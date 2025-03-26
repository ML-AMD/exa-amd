import os
import logging
import time

from tools.errors import VaspNonReached
from parsl.app.errors import AppTimeout
from parsl.app.errors import BashExitFailure

from tools.config_manager import ConfigManager


def vasp_calculations(config):
    from parsl_tasks.dft_optimization import run_vasp_calc
    from parsl_tasks.dft_optimization import fused_vasp_calc_init_perf
    work_dir = config["work_dir"]
    output_file_vasp_calc = os.path.join(
        config["vasp_work_dir"], config["output_file"])

    fused_vasp_calc_init_perf().result()
    exec_time = -1
    # open the output file to log the structures that failed or succeded to converge
    fp = open(output_file_vasp_calc, 'w')
    fp.write("id,result\n")

    # launch all vasp calculations
    start_dft_calc = time.time()
    exec_time = -time.time()
    l_futures = [run_vasp_calc(config.get_json_config(), i)
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
            logging.warning(f"An exception occurred: {e}")
            fp.write("{},{}\n".format(id, "unexpected_error"))

    fp.close()
    exec_time += time.time()
    end_dft_calc = time.time()
    return exec_time


def generate_structures(config):
    from parsl_tasks.gen_structures import gen_structures
    from parsl_tasks.gen_structures import gen_structures_init_perf
    exec_time = -1
    try:
        gen_structures_init_perf().result()
        exec_time = -time.time()
        gen_structures(config.get_json_config()).result()
        exec_time += time.time()
    except Exception as e:
        logging.critical(f"An exception occurred: {e}")
    return exec_time


def select_structures(config):
    from parsl_tasks.select_structures import select_structures
    from parsl_tasks.select_structures import select_structures_init_perf
    exec_time = -1
    try:
        select_structures_init_perf().result()
        exec_time = -time.time()
        select_structures(config.get_json_config()).result()
        exec_time += time.time()
    except Exception as e:
        logging.critical(f"An exception occurred: {e}")
    return exec_time


def run_cgcnn(config):
    from parsl_tasks.cgcnn import cgcnn_prediction
    from parsl_tasks.cgcnn import cgcnn_prediction_init_perf
    exec_time = -1
    try:
        cgcnn_prediction_init_perf().result()
        exec_time = -time.time()
        cgcnn_prediction(config.get_json_config()).result()
        exec_time += time.time()
    except Exception as e:
        logging.critical(f"An exception occurred: {e}")
    return exec_time


def run_workflow(config):
    logging.info("Start the 'vasp_based' workflow'")
    time_gen_struct = -1
    if not os.path.exists(os.path.join(config["work_dir"], 'structures/1.cif')):
        time_gen_struct = generate_structures(config)
    logging.info(f"generate_structures done: {time_gen_struct}")

    time_cgcnn = -1
    if not os.path.exists(os.path.join(config["work_dir"], 'test_results.csv')):
        time_cgcnn = run_cgcnn(config)
    logging.info(f"cgcnn done: {time_cgcnn}")

    time_select_struct = -1
    if not os.path.exists(os.path.join(config["work_dir"], 'new/POSCAR_1')):
        time_select_struct = select_structures(config)
    logging.info(f"select structures done {time_select_struct}")

    config.setup_vasp_calculations()
    time_vasp = vasp_calculations(config)
    logging.info(f"vasp calculations done {time_vasp}")
