import os, torch, random, cv2, glob
import numpy as np
from torch.utils import data

class TrueVideoDataset(data.Dataset):
    """Data Generator inherited from keras.utils.Sequence
    Args:
        directory: the path of data set, and each sub-folder will be assigned to one class
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """

    def __init__(self, directory, target_frames=64, sample='uniform_sampling', gap=2):
        # Initialize the params
        self.directory = directory
        self.target_frames = target_frames
        self.sample = sample
        self.gap = gap  # only for gap sampling
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.X_path = self.search_data()
        # Print basic statistics information
        self.print_stats()

    def search_data(self):
        X_path = []
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        for i, folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory, folder)
            if os.path.isfile(folder_path):
                continue
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # append the each file path, and keep its label
                X_path.append(file_path)
        return X_path

    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_dirs = len(self.dirs)
        # Output states
        print("Found {} files belonging to {} dirs.".format(self.n_files, self.n_dirs))

    def __len__(self):
        # calculate the iterations of each epoch
        return len(self.X_path)

    def __getitem__(self, index):
        path = self.X_path[index]
        # get batch data
        x = self.data_generation(path)
        return x

    def data_generation(self, path):
        # load data into memory, you can change the np.load to any method you want
        x = self.load_data(path)
        # transfer the data format and take one-hot coding for labels
        x = np.array(x)
        return x

    def uniform_sampling(self, video, target_frames=64):
        # get total frames of input video and calculate sampling interval
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames / target_frames))
        # init empty list for sampled video and
        sampled_video = []
        for i in range(0, len_frames, interval):
            sampled_video.append(video[i])
            # calculate numer of padded frames and fix it
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad > 0:
            for i in range(-num_pad, 0):
                try:
                    padding.append(video[i])
                except:
                    padding.append(video[0])
            sampled_video += padding
            # get sampled video
        return np.array(sampled_video, dtype=np.float32)

    def random_continuous_sampling(self, video, target_frames=64):
        start_point = np.random.randint(len(video) - target_frames)
        return video[start_point:start_point + target_frames]

    def random_gap_sampling(self, video, target_frames=64, gap=2):
        start_point = np.random.randint(len(video) - (target_frames - 1) * gap)
        return video[start_point:start_point + (target_frames - 1) * gap + 1:gap]

    def load_data(self, path):
        # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
        data = np.load(path)
        if self.sample == 'uniform_sampling':
            # sampling frames uniformly from the entire video
            data = self.uniform_sampling(video=data, target_frames=self.target_frames)
        elif self.sample == 'random_continuous_sampling':
            # sampling target number frames continuously from the entire video
            data = self.random_continuous_sampling(video=data, target_frames=self.target_frames)
        elif self.sample == 'random_gap_sampling':
            # sample target number frames by gap from the entire video
            data = self.random_gap_sampling(video=data, target_frames=self.target_frames, gap=self.gap)
        elif self.sample == 'no_sampling':
            data = data
        return data