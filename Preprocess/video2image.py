import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def video2image(file_dir, save_dir):
    """Transfer all the videos and save them into specified directory
    Args:
        file_dir: source folder of target videos
        save_dir: destination folder of output .npy files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # List the files
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        video_name = v.split('.')[0]
        video_path = os.path.join(file_dir, v)
        cap = cv2.VideoCapture(video_path)
        len_frames = int(cap.get(7))
        assert len_frames == 150, "frame num is not 150"
        image_dir = os.path.join(save_dir, video_name)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        print('processing ', video_path, ' frames: ', len_frames)
        try:
            for i in range(len_frames):
                _, frame = cap.read()
                cv2.imwrite(os.path.join(image_dir, '%03g.jpg'%i), frame)
        except:
            print("Error: ", video_path, len_frames, i)
        finally:
            cap.release()

    return None

def videos2image(source_path, target_path):
    for f1 in ['train', 'val']:
        for f2 in ['Fight', 'NonFight']:
            path1 = os.path.join(source_path, f1, f2)
            path2 = os.path.join(target_path, f1, f2)
            video2image(file_dir=path1, save_dir=path2)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser('RWF baseline violence detect model.')
    parser.add_argument('-source_path', type=str, default='/home/lwg/workspace/Dataset/RWF-2000/RWF-2000', help='rwf video source path')
    parser.add_argument('-target_path', type=str, default='/home/lwg/workspace/Dataset/RWF-2000/RWF-2000-images',
                        help='rwf .npy files path')
    args = parser.parse_args()

    source_path = args.source_path
    target_path = args.target_path
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    videos2image(source_path, target_path)