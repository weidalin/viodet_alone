import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from video2numpy_withflow import Save2Npy

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser('RWF baseline violence detect model.')
    parser.add_argument('-source_path', type=str, default='/home/weida/workspace/Dataset/realdata/', help='rwf video source path')
    parser.add_argument('-target_path', type=str, default='/home/weida/workspace/Dataset/realdata/realdata-npy-withflow',
                        help='rwf .npy files path')
    args = parser.parse_args()

    source_path = args.source_path
    target_path = args.target_path

    Save2Npy(file_dir=source_path, save_dir=target_path)
