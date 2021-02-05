import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def Video2Npy(file_path, resize=(224, 224)):
    """Load video and tansfer it into .npy format
    Args:
        file_path: the path of video file
        resize: the target resolution of output video
    Returns:
        frames: gray-scale video
        flows: magnitude video of optical flows
    """
    # Load video
    cap = cv2.VideoCapture(file_path)
    # Get number of frames
    len_frames = int(cap.get(7))
    # Extract frames from video
    try:
        frames = []
        for i in range(len_frames - 1):
            _, frame = cap.read()
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224, 224, 3))
            frames.append(frame)
    except:
        print("Error: ", file_path, len_frames, i)
    finally:
        frames = np.array(frames)
        cap.release()

    # Get the optical flow of video
    # flows = getOpticalFlow(frames)

    result = np.zeros((len(frames), 224, 224, 3))
    result[..., :3] = frames
    # result[...,3:] = flows

    return result


def Save2Npy(file_dir, save_dir):
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
        # Get dest
        save_path = os.path.join(save_dir, video_name + '.npy')
        # Load and preprocess video
        data = Video2Npy(file_path=video_path, resize=(224, 224))
        data = np.uint8(data)
        # Save as .npy file
        np.save(save_path, data)

    return None

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser('RWF baseline violence detect model.')
    parser.add_argument('-source_path', type=str, default='/home/weida/workspace/Dataset/RWF-2000/RWF-2000', help='rwf video source path')
    parser.add_argument('-target_path', type=str, default='/home/weida/workspace/Dataset/RWF-2000/RWF-2000-npy-rgbonly',
                        help='rwf .npy files path')
    args = parser.parse_args()

    source_path = args.source_path
    target_path = args.target_path

    for f1 in ['train', 'val']:
        for f2 in ['Fight', 'NonFight']:
            path1 = os.path.join(source_path, f1, f2)
            path2 = os.path.join(target_path, f1, f2)
            Save2Npy(file_dir=path1, save_dir=path2)