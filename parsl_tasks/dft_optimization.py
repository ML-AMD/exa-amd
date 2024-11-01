from parsl import python_app, bash_app

@bash_app(executors=['chicoma_single_gpu_per_worker'])
def vasp_relaxation(work_dir, id, ):
    import os 
    import shutil
    os.chdir(work_dir)

    output_file = os.path.join(work_dir,"output_{}.rx".format(id))
    vasp_std_exe = "/usr/projects/icapt/applications/vasp/vasp-6.4.2-nvidia-gpu/bin/vasp_std"
    poscar = os.path.join("/users/moraru/Parsl-Project/data/select_structures/new", "POSCAR_{}".format(id))
    incar = "/users/moraru/Parsl-Project/CMS/ctest/INCAR.rx"
    potcar = "/users/moraru/Parsl-Project/data/POTCAR"    
    
    shutil.copy(poscar, os.path.join(work_dir, "POSCAR"))
    shutil.copy(incar, os.path.join(work_dir, "INCAR"))
    shutil.copy(potcar, os.path.join(work_dir,"POTCAR"))
    
    return "timeout 10 $PARSL_SRUN_PREFIX {} > {} ".format(vasp_std_exe, output_file)