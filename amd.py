import sys
from parsl_configs.perlmutter import exec_config_debug
from tools.errors import VaspNonReached
from parsl.app.errors import AppTimeout
from parsl.app.errors import BashExitFailure
import parsl
import time
import os
import json
    
parsl.load(exec_config_debug)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def vasp_calculations(config):
    from parsl_tasks.dft_optimization import run_vasp_calc
    output_file_vasp_calc = config["output_file_name"]
    work_dir = config["work_dir"]
    work_subdir_prefix = "work_subdir"
    nb_of_dft_calculations = 4
    
    start_dft_calc = time.time()
    # launch the tasks (all the vasp calculations)
    l_futures = []
    for i in range(1, nb_of_dft_calculations+1):
        work_subdir = os.path.join(work_dir,"{}_{}".format(work_subdir_prefix,i))
        if not os.path.exists(work_subdir):
            os.makedirs(work_subdir)
            l_futures.append(run_vasp_calc(config, work_subdir, i))
        else:
            eprint("work_dir ({}) already exists".format(work_dir))
            sys.exit(1)
            
    # open the output file to log the structures that failed or succeded to converge
    fp = open(output_file_vasp_calc, 'w')
    fp.write("id,result\n")
    
    # wait for all the tasks to complete
    for future, id in l_futures:
        try:
            err = future.exception()
            if err:
                raise err
            fp.write("{},{}\n".format(id,"success"))
        except VaspNonReached:
            fp.write("{},{}\n".format(id,"non_reached"))
        except AppTimeout:
            fp.write("{},{}\n".format(id,"time_out"))
        except BashExitFailure:
            fp.write("{},{}\n".format(id,"bash_exit_failure"))
        except Exception as e:
            eprint("Exception: ", e)
            fp.write("{},{}\n".format(id,"unexpected_error"))
    
    fp.close()
    end_dft_calc = time.time()
    print("Elapsed time : {}".format(end_dft_calc - start_dft_calc))

def generate_structures(config):
    from parsl_tasks.gen_structures import gen_structures
    try:
        gen_structures(config).result()
    except Exception as e:
        eprint("Exception: ", e)
    

def select_structures(config):
    from parsl_tasks.select_structures import select_structures
    f = select_structures(config)
    try:
        f.result()
    except Exception as e:
        eprint("Exception: ", e)
        
def run_cgcnn(config):
    from parsl_tasks.cgcnn import cgcnn_prediction
    f = cgcnn_prediction(config)
    try:
        f.result()
    except Exception as e:
        eprint("Exception: ", e)

if __name__ == '__main__':
    config_file_name = "configs/std_config.json"
    with open(config_file_name, "r") as file:
        config = json.load(file)
    
    work_dir = config["work_dir"]
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
     
    generate_structures(config)
    print("-- generate_structures done...")
    run_cgcnn(config)
    print("-- run_cgcnn done...")
    select_structures(config)
    print("-- select_structures done...")
    vasp_calculations(config)