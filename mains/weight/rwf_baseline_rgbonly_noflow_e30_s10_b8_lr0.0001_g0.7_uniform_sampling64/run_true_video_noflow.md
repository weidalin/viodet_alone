
# command of trained this model:

    python rwf_train.py -data_name "RWF-2000-npy-noflow" -gpu "0, 1" -b 8 -lr 1e-4

# infer true video of a path: /home/weida/workspace/Dataset/realdata/
## step 1: preprocess all video of a path

    #cd mains
    #old version
    #python video2numpy_noflow_true_video_floder.py -source_path [] -target_path []
    #python true_video_split.py -source_path /home/weida/workspace/Dataset/realdata/violence1.avi -target_path /home/weida/workspace/Dataset/realdata-npy-noflow

    python true_video_split.py -source_dir [SOURCE_DIR] -target_dir [TARGET_DIR] -windows [WINDOW Legnth] -interval [Sliding stride]  
for example
  
     python true_video_split.py -source_dir "/home/weida/workspace/Dataset/realdata/violence1.avi" -target_dir "/home/weida/workspace/Dataset/realdata-npy-noflow"
    

## step 2: use this model detect violence of real data offline

    #cd mains
    python true_video_detection.py -model_type[] -test_dir[] -model_path[] -checkpoint[] ...
for example
    
    #path_1: /home/weida/workspace/Dataset/RWF-2000/RWF-2000-npy-noflow-2D/true/Fight
    #path_2: /home/weida/workspace/Dataset/RWF-2000/RWF-2000-npy-noflow-2D/true/NonFight
    #path_3: /home/weida/workspace/Dataset/RWF-2000/RWF-2000-npy-noflow/true/Fight
    
    #train_data_np_path: /home/weida/workspace/Dataset/RWF-2000/RWF-2000-npy-noflow

    python true_video_detection.py -model_type rgbonly -test_dir /home/weida/workspace/Dataset/RWF-2000/RWF-2000-npy-noflow -model_path weight/rwf_baseline_rgbonly_noflow_e30_s10_b8_lr0.0001_g0.7_uniform_sampling64 -checkpoint epoch11_0.77.pth
    

## step 3: check result 

    cat /home/weida/workspace/Dataset/realdata-npy-noflow/rwf_baseline_rgbonly_noflow_e30_s10_b8_lr0.0001_g0.7_uniform_sampling64_epoch11_0.77_true_video_res.txt
