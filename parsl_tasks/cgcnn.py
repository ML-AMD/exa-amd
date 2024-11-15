from parsl import python_app, bash_app

@bash_app(executors=['single_gpu_per_worker'])
def cgcnn_prediction(config):
    import os 
    import shutil
    try:
        os.chdir(config["work_dir"])
    
        predict_script_path = config["scripts"]["cgcnn_predict"]
        model_path = config["model"]["cgcnn_model"]

        dir_structures = os.path.join(config["work_dir"], "structures") # @@@ structures
        csv_id_prop = os.path.join(config["data_dir_path"], "id_prop.csv")
        atom_init_json = os.path.join(config["data_dir_path"], "atom_init.json")
        
        shutil.copy(csv_id_prop, dir_structures)
        shutil.copy(atom_init_json, dir_structures)
    except Exception as e:
        raise e
    
    return "python {} {} {} --batch-size 256 --workers 32 ".format(predict_script_path, model_path, dir_structures)