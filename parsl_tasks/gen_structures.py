from parsl import bash_app

@bash_app(executors=['cpu_single_node'])
def gen_structures(config):
    import os 
    try:
        dir_structures = os.path.join(config["work_dir"], "structures") # @@@ structures
        dir_mp_structures = os.path.join(config["data_dir_path"], "mpstrs")
        csv_elements = os.path.join(config["data_dir_path"], "Paul_nomix_3pair.csv")
        
        os.makedirs(dir_structures)
        os.chdir(dir_structures)
    except Exception as e:
        raise e
    
    return "python {} --num_workers 256 --input_dir {} --input_csv_elements {}".format(config["scripts"]["gen_structure"], dir_mp_structures, csv_elements)