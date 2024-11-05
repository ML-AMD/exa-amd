import sys
from parsl_configs.chicoma import chicoma_config
from parsl_tasks.dft_optimization import run_vasp_calc
import parsl
import time
import os
import json
    
parsl.load(chicoma_config)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == '__main__':
    
    config_file_name = "/users/moraru/Parsl-Project/exa-amd/configs/std_config.json"
    with open(config_file_name, "r") as file:
        config = json.load(file)

    output_file_vasp_calc = config["output_file_name"]
    work_dir = config["work_dir"]
    work_subdir_prefix = "work_subdir"
    nb_of_dft_calculations = 4
    
    start_dft_calc = time.time()
    # launch the tasks (all the vasp calculations)
    l_futures = []
    for i in range( 204, nb_of_dft_calculations+204):
        work_subdir = os.path.join(work_dir,"{}_{}".format(work_subdir_prefix,i))
        if not os.path.exists(work_subdir):
            os.makedirs(work_subdir)
            l_futures.append(run_vasp_calc(config_file_name, work_subdir, i))
        else:
            eprint("work_dir ({}) already exists".format(work_dir))
            sys.exit(1)
            
    # open the output file to log the structures that failed or succeded to converge
    fp = open(output_file_vasp_calc, 'w')
    fp.write("id,result\n")
    
    # wait for all the tasks to complete
    for future in l_futures:
        try:
            id, result = future.result()
            fp.write("{},{}\n".format(id,result))
            
        except Exception as e:
            eprint("Exception: ", e)
    
    end_dft_calc = time.time()
    print("Elapsed time : {}".format(end_dft_calc - start_dft_calc))