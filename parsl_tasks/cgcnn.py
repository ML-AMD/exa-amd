from parsl import python_app, bash_app

#@bash_app(executors=['chicoma_gpu_single_node'])
def cgcnn_prediction(iter):
    import os 
    prefix_cgcnn = "/users/moraru/Parsl-Project/CMS/ctest/"
    predict_script_path = prefix_cgcnn + "predict.py"
    model_path = prefix_cgcnn + "form_1st.pth.tar"
    data_path =prefix_cgcnn + "generated_structures"
    
    return "python {} {} {} --batch-size 256 --workers ".format(predict_script_path, model_path, data_path)