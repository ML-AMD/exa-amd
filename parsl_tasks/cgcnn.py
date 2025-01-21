from parsl import python_app, bash_app


@bash_app(executors=['single_gpu_per_worker_cgcnn'])
def cgcnn_prediction(config):
    import os
    import shutil
    try:
        os.chdir(config["work_dir"])

        predict_script_path = os.path.join(config["cms_dir"], "predict.py")
        model_path = os.path.join(config["cms_dir"], "form_1st.pth.tar")

        dir_structures = os.path.join(
            config["work_dir"], "structures")  # @@@ structures
        atom_init_json = os.path.join(config["cms_dir"], "atom_init.json")

        # shutil.copy(csv_id_prop, dir_structures)
        shutil.copy(atom_init_json, dir_structures)
    except Exception as e:
        raise

    return "python {} {} {} --batch-size {} --workers {} ".format(predict_script_path, model_path, dir_structures, config["batch_size"], config["num_workers"])
