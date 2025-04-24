from parsl import python_app, bash_app
from parsl_configs.parsl_executors_labels import CGCNN_EXECUTOR_LABEL


@bash_app(executors=[CGCNN_EXECUTOR_LABEL])
def cgcnn_prediction(config):
    import os
    import shutil
    try:
        os.chdir(config["work_dir"])

        predict_script_path = os.path.join(config["cms_dir"], "predict.py")
        model_path = os.path.join(config["cms_dir"], "form_1st.pth.tar")

        dir_structures = os.path.join(
            config["work_dir"], "structures")
        atom_init_json = os.path.join(config["cms_dir"], "atom_init.json")

        shutil.copy(atom_init_json, dir_structures)
    except Exception as e:
        raise
    num_workers = config["num_workers"]
    return "python {} {} {} --batch-size {} --workers {} ".format(
        predict_script_path, model_path, dir_structures, config["batch_size"], num_workers)
