from parsl import bash_app

@bash_app(executors=['cpu_single_node'])
def select_structures(config):
    import os
    try:
        os.chdir(config["work_dir"])
        
        tr_csv_file = os.path.join(config["work_dir"], "test_results.csv") 
        dir_structures = os.path.join(config["work_dir"], "structures") # @@@ structures

    except Exception as e:
        raise e
    
    return "python {} --ef_threshold -0.2 --num_workers 256 --csv_file {} --nomix_dir {}".format(config["scripts"]["select_structure"], tr_csv_file, dir_structures)