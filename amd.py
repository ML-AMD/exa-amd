import sys
from parsl_configs.chicoma import chicoma_config
from parsl_tasks.dft_optimization import vasp_relaxation
import parsl
import time
import os

parsl.load(chicoma_config)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == '__main__':
    work_dir="/users/moraru/Parsl-Project/exa-amd/work_dir"
    work_subdir_prefix="tmp"
    nb_of_dft_calculations = 4
    
    start_dft_calc = time.time()
    l_futures = []
    for i in range(1, nb_of_dft_calculations+1):
        work_subdir = os.path.join(work_dir,"{}_{}".format(work_subdir_prefix,i))
        if not os.path.exists(work_subdir):
            os.makedirs(work_subdir)
            l_futures.append(vasp_relaxation(work_subdir, i))
        else:
            eprint("work_dir ({}) already exists".format(work_dir))
        
    for future in l_futures:
        try:
            future.result()
        except parsl.app.errors.BashExitFailure as e:
            eprint("Exception: ", e)
    
    end_dft_calc = time.time()
    print("Elapsed time : {}".format(end_dft_calc - start_dft_calc))