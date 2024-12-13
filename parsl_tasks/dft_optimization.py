from parsl import python_app, bash_app, join_app


@bash_app(executors=['single_gpu_per_worker'])
def vasp_relaxation(config, id, walltime=(int)):
    import os
    import shutil

    try:
        work_subdir = os.path.join(config["work_dir"], str(id))
        if not os.path.exists(work_subdir):
            os.makedirs(work_subdir)
        os.chdir(work_subdir)
        output_file = os.path.join(work_subdir, "output.rx")

        vasp_std_exe = config["vasp_std_exe"]
        poscar = os.path.join(config["work_dir"],
                              "new", "POSCAR_{}".format(id))
        incar = os.path.join(config["cms_dir"], "INCAR.rx")
        os.symlink(os.path.join(config["work_dir"], "POTCAR"), "POTCAR")

        # relaxation
        shutil.copy(poscar, os.path.join(work_subdir, "POSCAR"))
        shutil.copy(incar, os.path.join(work_subdir, "INCAR"))

        # Change NSW iterations
        FORCE_CONV = config["force_conv"]
        os.system(f"sed -i 's/NSW\\s*=\\s*[0-9]*/NSW = {FORCE_CONV}/' INCAR")
    except Exception as e:
        raise

    return "$PARSL_SRUN_PREFIX {} > {} ".format(vasp_std_exe, output_file)


@bash_app(executors=['single_gpu_per_worker'])
def vasp_energy_calculation(dependency_f, config, id, walltime=(int)):
    import os
    import shutil
    from tools.errors import VaspNonReached

    try:
        work_subdir = os.path.join(config["work_dir"], str(id))
        output_rx = os.path.join(work_subdir, "output.rx")

        # grep "reached"
        # reached = False
        FORCE_CONV = config["force_conv"]
        with open(output_rx, "r") as file:
            for line in file:
                if "reached" in line or "{FORCE_CONV} F=" in line:
                    reached = True
                    break

        if not reached:
            raise VaspNonReached

        os.chdir(work_subdir)

        vasp_std_exe = config["vasp_std_exe"]
        incar_en = os.path.join(config["cms_dir"], "INCAR.en")
        output_file = os.path.join(work_subdir, "output_{}.en".format(id))

        os.rename("OUTCAR", "OUTCAR_{}.rx".format(id))
        shutil.copy("CONTCAR", os.path.join(
            work_subdir, "CONTCAR_{}".format(id)))
        shutil.copy("CONTCAR", "POSCAR")
        shutil.copy(incar_en, "INCAR")

    except Exception as e:
        raise

    return "$PARSL_SRUN_PREFIX {} > {} ".format(vasp_std_exe, output_file)


@python_app(executors=['single_gpu_per_worker'])
def fused_vasp_calc(config, id, walltime=(int)):
    import os
    import shutil
    from tools.errors import VaspNonReached

    try:
        work_subdir = os.path.join(config["work_dir"], str(id))
        if not os.path.exists(work_subdir):
            os.makedirs(work_subdir)
        os.chdir(work_subdir)

        #
        # prepare relaxation
        #
        output_file = os.path.join(work_subdir, "output.rx")

        vasp_std_exe = config["vasp_std_exe"]
        poscar = os.path.join(config["work_dir"],
                              "new", "POSCAR_{}".format(id))
        incar = os.path.join(config["cms_dir"], "INCAR.rx")
        os.symlink(os.path.join(config["work_dir"], "POTCAR"), "POTCAR")

        # relaxation
        shutil.copy(poscar, os.path.join(work_subdir, "POSCAR"))
        shutil.copy(incar, os.path.join(work_subdir, "INCAR"))

        # Change NSW iterations
        FORCE_CONV = config["force_conv"]
        os.system(f"sed -i 's/NSW\\s*=\\s*[0-9]*/NSW = {FORCE_CONV}/' INCAR")

        # run relaxation
        srun_cmd = "timeout {} $PARSL_SRUN_PREFIX {} > {} ".format(
            config["walltime"], vasp_std_exe, output_file)
        os.system(srun_cmd)

        #
        # prepare energy calculation
        #
        output_rx = os.path.join(work_subdir, "output.rx")

        # grep "reached"
        reached = False
        FORCE_CONV = config["force_conv"]
        with open(output_rx, "r") as file:
            for line in file:
                if "reached" in line or "{FORCE_CONV} F=" in line:
                    reached = True
                    break
        if not reached:
            raise VaspNonReached

        incar_en = os.path.join(config["cms_dir"], "INCAR.en")
        output_file_en = os.path.join(work_subdir, "output_{}.en".format(id))

        os.rename("OUTCAR", "OUTCAR_{}.rx".format(id))
        shutil.copy("CONTCAR", os.path.join(
            work_subdir, "CONTCAR_{}".format(id)))
        shutil.copy("CONTCAR", "POSCAR")
        shutil.copy(incar_en, "INCAR")

        # run relaxation
        srun_cmd = "timeout {} $PARSL_SRUN_PREFIX {} > {} ".format(
            config["walltime"], vasp_std_exe, output_file_en)
        os.system(srun_cmd)

    except Exception as e:
        raise


def run_vasp_calc(config, id):
    # f_relax = vasp_relaxation(config, id, walltime=int(config["walltime"]))
    # f_energy = vasp_energy_calculation(f_relax, config, id, walltime=int(config["walltime"]))
    # return f_energy, id
    f_vasp = fused_vasp_calc(config, id, walltime=int(config["walltime"]))
    return f_vasp, id
