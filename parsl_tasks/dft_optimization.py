from parsl import python_app, bash_app, join_app
from tools.errors import VaspNonReached

@bash_app(executors=['single_gpu_per_worker'])
def vasp_relaxation(config, work_subdir, id, walltime=(int)):
    import os 
    import shutil
    try:
        os.chdir(work_subdir)
        output_file = os.path.join(work_subdir,"output_{}.rx".format(id))

        vasp_std_exe = config["vasp_std_exe"]
        poscar = os.path.join(config["work_dir"], "new", "POSCAR_{}".format(id))
        incar = os.path.join(config["data_dir_path"], "INCAR.rx")
        potcar = os.path.join(config["data_dir_path"], "POTCAR")
        

        # relaxation
        shutil.copy(poscar, os.path.join(work_subdir, "POSCAR"))
        shutil.copy(incar, os.path.join(work_subdir, "INCAR"))
        shutil.copy(potcar, os.path.join(work_subdir,"POTCAR"))
    except Exception as e:
        raise e
    
    return "$PARSL_SRUN_PREFIX {} > {} ".format(vasp_std_exe, output_file)


@bash_app(executors=['single_gpu_per_worker'])
def vasp_energy_calculation(dependency_f, onfig, work_subdir, id, walltime=(int)):
    import os 
    import shutil

    try:
        output_rx = os.path.join(work_subdir,"output_{}.rx".format(id))
        
        # grep "reached"
        reached = False
        with open(output_rx, "r") as file:
            for line in file:
                if "reached" in line:
                    reached = True
                    break
        
        if not reached:
            raise VaspNonReached  

        os.chdir(work_subdir)
            
        vasp_std_exe = config["vasp_std_exe"]
        incar_en = os.path.join(config["data_dir_path"], "INCAR.en")
        output_file = os.path.join(work_subdir,"output_{}.en".format(id))
        
        os.rename("OUTCAR", "OUTCAR_{}.rx".format(id))
        shutil.copy("CONTCAR", "CONTCAR_{}".format(id))
        shutil.copy("CONTCAR", "POSCAR")
        shutil.copy(incar_en, "INCAR")
        
    except Exception as e:
        raise e
    
    return "$PARSL_SRUN_PREFIX {} > {} ".format(vasp_std_exe, output_file)


def run_vasp_calc(config, work_subdir, id):
    f_relax: Future = vasp_relaxation(config, work_subdir, id, walltime=3600)
    f_energy: Future = vasp_energy_calculation(f_relax, config, work_subdir, id, walltime=3600)
    return f_energy, id
