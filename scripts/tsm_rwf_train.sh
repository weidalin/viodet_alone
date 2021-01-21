#!/usr/bin/env bash
# python ../mains/tsm_rwf_train.py rwf RGB --arch mobilenetv2 --num_segments 16 --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 --batch-size 8 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb
python ../mains/rwf_train.py -data_name "RWF-2000-npy-noflow" -gpu "1" -b 8 -lr 1e-4 
python ../mains/rwf_train.py -data_name "RWF-2000-npy-noflow" -gpu "1" -b 8 -lr 1e-4 
python ../mains/rwf_train.py -data_name "RWF-2000-npy-noflow" -gpu "1" -b 8 -lr 1e-4 
python ../mains/rwf_train.py -data_name "RWF-2000-npy-noflow" -gpu "1" -b 8 -lr 1e-4 
