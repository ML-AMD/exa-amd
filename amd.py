import sys
from parsl_configs.perlmutter import exec_config_debug
from parsl_configs.chicoma import chicoma_config
from tools.errors import VaspNonReached
from parsl.app.errors import AppTimeout
from parsl.app.errors import BashExitFailure
import parsl
import time
import os
import json
import math
import re
    
parsl.load(chicoma_config)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def find_next(work_dir):
    # Get current directory contents
    contents = os.listdir(work_dir)

    # Filter only directories and find those that match numeric pattern
    numbered_dirs = []
    for item in contents:
        if os.path.isdir(os.path.join(work_dir,item)):
            # Try to extract a number from the directory name
            match = re.search(r'^\d+', item)
            if match:
                numbered_dirs.append(int(match.group()))

    if not numbered_dirs:
        return 1  # If no numbered directories exist, start with 1

    # Find the highest number and add 1
    next_number = max(numbered_dirs) + 1
    return next_number

def vasp_calculations(config):
    from parsl_tasks.dft_optimization import run_vasp_calc
    work_dir = config["work_dir"]
    output_file_vasp_calc = os.path.join(work_dir,config["output_file"])
    num_strs=config["num_strs"]
    total_workers=config["num_vasp_jobs"]*config["num_gpu_nodes"]*4
    nstart=find_next(work_dir)
    eprint("Start structure: ", nstart)

    # Calculate batch size and number of batches
    batch_size = total_workers
    num_batches = math.ceil(num_strs / batch_size)

    start_dft_calc = time.time()

    num_batches = 1 # comment MM
    # Process structures in batches
    for batch in range(num_batches):
        batch_start = batch * batch_size + nstart
        batch_end = min((batch + 1) * batch_size, num_strs) + nstart
        batch_end = batch_start + 1 # comment MM 
        # Launch batch of tasks
        l_futures = []
        for i in range(batch_start, batch_end):
            print("run task {}".format(i))
            #work_subdir = os.path.join(work_dir, "{}".format(i))
            #if not os.path.exists(work_subdir):
            #    os.makedirs(work_subdir)
            l_futures.append(run_vasp_calc(config, i))
            #else:
            #    eprint("work_dir ({}) already exists".format(work_subdir))
            #    continue
           
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
        eprint("Exception [{}]: ".format(type(e).__name__), e)
        sys.exit(1)
    

def select_structures(config):
    from parsl_tasks.select_structures import select_structures
    f = select_structures(config)
    try:
        f.result()
    except Exception as e:
        eprint("Exception: ", e)
        sys.exit(1)
        
def run_cgcnn(config):
    from parsl_tasks.cgcnn import cgcnn_prediction
    f = cgcnn_prediction(config)
    try:
        f.result()
    except Exception as e:
        eprint("Exception: ", e)
        sys.exit(1)

if __name__ == '__main__':
    config_file_name = "configs/chicoma.json"
    with open(config_file_name, "r") as file:
        config = json.load(file)
    
    work_dir = os.path.join(config["work_dir"],config["elements"])
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    #Update work_dir
    config["work_dir"]=work_dir
     
    if not os.path.exists(os.path.join(work_dir,'structures/1.cif')):
       generate_structures(config)
    print("-- generate_structures done...")

    if not os.path.exists(os.path.join(work_dir,'test_results.csv')):
       run_cgcnn(config)
    print("-- run_cgcnn done...")

    if not os.path.exists(os.path.join(work_dir,'new/POSCAR_1')):
       select_structures(config)
    print("-- select_structures done...")

    # Create POTCAR file
    POTDIR=config['pot_dir']
    ele1,ele2,ele3=config["elements"].split('-')
    potcar_command = f"cat {POTDIR}/{ele1}/POTCAR {POTDIR}/{ele2}/POTCAR {POTDIR}/{ele3}/POTCAR > {work_dir}/POTCAR"
    os.system(potcar_command)

    # # Count Structures for VASP calculations

    structure_files = [f for f in os.listdir(os.path.join(work_dir,"new")) if f.startswith("POSCAR_")]
    num_structures = len(structure_files)
    config["num_strs"]=num_structures

    vasp_calculations(config)
