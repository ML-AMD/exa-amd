from parsl import python_app, bash_app, join_app
from tools.errors import VaspNonReached

@python_app
def const_future(id, msg):
    return (id, msg)

@bash_app(executors=['single_gpu_per_worker'])
def vasp_relaxation(config_file_name, work_dir, id, walltime=(int)):
    import os 
    import shutil
    import json
    os.chdir(work_dir)
    with open(config_file_name, "r") as file:
        config = json.load(file)
    
    output_file = os.path.join(work_dir,"output_{}.rx".format(id))
    vasp_std_exe = config["vasp_std_exe"]
    poscar = os.path.join(config["selected_structures_dir"], "POSCAR_{}".format(id))
    incar = config["incar_rx"]
    potcar = config["potcar"] 
    
    # relaxation
    shutil.copy(poscar, os.path.join(work_dir, "POSCAR"))
    shutil.copy(incar, os.path.join(work_dir, "INCAR"))
    shutil.copy(potcar, os.path.join(work_dir,"POTCAR"))
    
    return "$PARSL_SRUN_PREFIX {} > {} ".format(vasp_std_exe, output_file)


@bash_app(executors=['single_gpu_per_worker'])
def vasp_energy_calculation(config_file_name, work_dir, id, walltime=(int)):
    import os 
    import shutil
    import json
    output_rx = os.path.join(work_dir,"output_{}.rx".format(id))
    
    # grep "reached"
    reached = False
    with open(output_rx, "r") as file:
        for line in file:
            if "reached" in line:
                reached = True
                break
    
    if not reached:
        raise VaspNonReached  

    os.chdir(work_dir)
    with open(config_file_name, "r") as file:
        config = json.load(file)
        
    vasp_std_exe = config["vasp_std_exe"]
    incar_en = config["incar_en"]
    output_file = os.path.join(work_dir,"output_{}.en".format(id))
    os.rename("OUTCAR", "OUTCAR_{}.rx".format(id))
    shutil.copy("CONTCAR", "CONTCAR_{}".format(id))
    shutil.copy("CONTCAR", "POSCAR")
    shutil.copy(incar_en, "INCAR")
    
    return "$PARSL_SRUN_PREFIX {} > {} ".format(vasp_std_exe, output_file)
    
@join_app()
def run_vasp_calc(config_file_name, work_dir, id):
    from parsl.app.errors import AppTimeout
    f_relax: Future = vasp_relaxation(config_file_name, work_dir, id, walltime=3600)
    try:
        f_relax.result()
    except AppTimeout as e:
        return const_future(id, "time_out")
    except parsl.app.errors.BashExitFailure as e:
        return const_future(id, "bash_exit_failure")
    except Exception as e:
        raise e 
    
    f_energy: Future = vasp_energy_calculation(config_file_name, work_dir, id, walltime=3600)
    try:
        f_energy.result()
    except VaspNonReached:
        return const_future(id, "non_reached")
    except AppTimeout as e:
        return const_future(id, "time_out")
    except parsl.app.errors.BashExitFailure as e:
        return const_future(id, "bash_exit_failure")
    except Exception as e:
        raise e
    
    return const_future(id, "success")
