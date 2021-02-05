import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def Save2Npy(file_dir, save_dir, precut_len=25, resize=(224, 224)):
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
        # Split video name
        video_name = v.split('.')[0]
        # Get src
        video_path = os.path.join(file_dir, v)
        # Load video
        cap = cv2.VideoCapture(video_path)
        # Get number of frames
        len_frames = int(cap.get(7))
        assert len_frames == 150, "frame num is not 150"
        # Extract frames from video
        npy_num = 0
        # result = np.zeros((25, 224, 224, 3))
        try:
            frames = []
            for i in range(len_frames):
                _, frame = cap.read()
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.reshape(frame, (224, 224, 3))
                frames.append(frame)
                if len(frames) == precut_len:
                    data = np.array(frames)
                    # result[..., :3] = data
                    save_path = os.path.join(save_dir, video_name + '_' + str(npy_num) + '.npy')
                    # Load and preprocess video
                    result = np.uint8(result)
                    # Save as .npy file
                    np.save(save_path, result)
                    frames.clear()
                    npy_num += 1
        except:
            print("Error: ", video_path, len_frames, i)
        finally:
            print(video_path, len_frames, ' be cut to ', npy_num, ' snippets by ', precut_len)
            cap.release()

    return None

def data_precut(source_path, target_path, precut_len=25):
    for f1 in ['train', 'val']:
        for f2 in ['Fight', 'NonFight']:
            path1 = os.path.join(source_path, f1, f2)
            path2 = os.path.join(target_path, f1, f2)
            Save2Npy(file_dir=path1, save_dir=path2, precut_len=precut_len)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser('RWF baseline violence detect model.')
    parser.add_argument('-source_path', type=str, default='/home/lwg/workspace/Dataset/RWF-2000/RWF-2000', help='rwf video source path')
    parser.add_argument('-target_path', type=str, default='/home/lwg/workspace/Dataset/RWF-2000/RWF-2000-npy-noflow-25',
                        help='rwf .npy files path')
    parser.add_argument('-precut_len', type=int, default=25, help='len of pre cut snippets')
    args = parser.parse_args()

    source_path = args.source_path
    target_path = args.target_path
    precut_len = args.precut_len
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    data_precut(source_path, target_path, precut_len)