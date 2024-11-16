from parsl import bash_app

@bash_app(executors=['cpu_single_node'])
def gen_structures(config):
    import os 
    try:
        dir_structures = os.path.join(config["work_dir"], "structures") # @@@ structures
        dir_mp_structures = os.path.join(config["cms_dir"], "mpstrs")
        dir_gen_structures = os.path.join(config["cms_dir"], "gen_structure.py")
        
        if not os.path.exists(dir_structures):
           os.makedirs(dir_structures)
        os.chdir(dir_structures)
    except Exception as e:
        raise e
    
    return "python {} --num_workers {} --input_dir {} --elements {}".format(dir_gen_structures, config["num_workers"], dir_mp_structures, config["elements"])
