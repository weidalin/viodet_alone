# ViolenceDetection
## RWF Dataset  
Follow Project [RWF2000-Video-Database-for-Violence-Detection](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)  
We change code from .ipynb to .ny, and provide both Keras and Pytorch codes

### Data Preprocess
For both keras and pytorch, transfer videos to .npys  

    cd Preprocess
RGB only(on flow):

    python video2numpy_noflow.py -source_path [] -target_path []

RGB+Flow(with flow):

    pyhton video2numpy_withflow.py -source_path [] -target_path []

Data shape: 3 channels for RGBonly .npys and 5 channels for RGB+flow .npys, the [:3] three are RGB and [3:] two are flow  
Datas have save in _/home/lwg/workspace/Dataset/RWF-2000/RWF-2000-npy-noflow_ and _/home/lwg/workspace/Dataset/RWF-2000/RWF-2000-npy-withflow_, which can be_used directly

### Keras model
For RGB only baseline
    
    python model/rgb_only_keras.py
    
For flowgate baseline
    
    python model/flow_gated_network_keras.py
  
### Pytorch model
Dataset:  
1, We provide four sampling method: _uniform sampling_, _random sampling_, _random gap sampling_ and _no sampling_  
2, We provide a normal rwf dataset class _dataset/rwf2000_dataset.py_, the _dataset/rwf2000_dataset2.py_ loads all data into memory once which can use the GPU as most as possible but cost memory resource. The former one is default.  
RWF Train :
    
    cd mains
    python rwf_train.py ...

The arguments of _rwf_train.py_ is described in the code.
For example, train baseline rgb-only model

    python rwf_train.py -model_type rgbonly -sample uniform_sampling -target_frames 64 -b 8
   
and train baseline flowgate model

    python rwf_train.py -model_type flowgate -sample uniform_sampling -target_frames 64 -b 8
    
It is worth to note that the Pytorch models is different with Keras in some details like model initialize, BN leyers, ReLU layers and activation functions, which can be adjusted in _model/rwf2000_baseline_rgbonly_ and _model/rwf2000_baseline_flowgate.py_ respectively.  
Models with trained weights and logs can be found in _mains/weight_ in the server, all of that can be used to inference.  
RWF Val:  
We can use the trained weights in _mains/weight_ to validate by

    python rwf_test.py ... 

For example, validate rgbpnly model and flowgate model

    python rwf_test.py -model_path weight/rwf_baseline_rgbonly_noflow__e30_s10_b8_lr0.01_g0.7 -checkpoint epoch28_0.8525.pth
    python rwf_test.py  -model_type flowgate -model_path weight/rwf_baseline_flowgate_withflow__e30_s10_b8_lr0.01_g0.7_uniform_sampling64 -checkpoint epoch12_0.8475.pth

## Real Data
### Data Preprocess
To split videos into snippets and save as .npy file  
    
    python true_video_split.py -source_dir [SOURCE_DIR] -target_dir [TARGET_DIR] -windows [WINDOW Legnth] -interval [Sliding stride]  
    
To detect violence of real data offline
    
    python true_video_detection.py -model_type[] -test_dir[] -model_path[] -checkpoint[] ...
For example
    
    python true_video_detection.py -test_dir /home/lwg/workspace/Dataset/True_Data_npy_150_75/ -model_path weight/rwf_baseline_rgbonly_noflow__e30_s10_b8_lr0.01_g0.7/ -checkpoint epoch28_0.8525.pth
    
To train 2D models in RWF Dataset

    python rwf_2Dnets.py