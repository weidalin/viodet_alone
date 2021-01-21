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

    def __init__(self, directory, data_augmentation=True, target_frames=64, sample='uniform_sampling', gap=2):
        # Initialize the params
        self.directory = directory
        self.data_aug = data_augmentation
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
            if os.path.isfile(folder_path) and folder_path.endswith(".npy"):
                file_path = folder_path
                # append the each file path, and keep its label
                X_path.append(file_path)
                continue
            # if os.path.isfile(folder_path):
            #     continue
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

    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video

    def uniform_sampling(self, video, target_frames=64):
        # get total frames of input video and calculate sampling interval
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames / target_frames))
        # init empty list for sampled video and
        sampled_video = []
        print("----------------len_frames", len_frames)
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

    def dynamic_crop(self, video):
        # extract layer of optical flow from video
        opt_flows = video[..., 3]
        # sum of optical flow magnitude of individual frame
        magnitude = np.sum(opt_flows, axis=0)
        # filter slight noise by threshold
        thresh = np.mean(magnitude)
        magnitude[magnitude < thresh] = 0
        # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
        x_pdf = np.sum(magnitude, axis=1) + 0.001
        y_pdf = np.sum(magnitude, axis=0) + 0.001
        # normalize PDF of x and y so that the sum of probs = 1
        x_pdf /= np.sum(x_pdf)
        y_pdf /= np.sum(y_pdf)
        # randomly choose some candidates for x and y
        x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
        y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)
        # get the mean of x and y coordinates for better robustness
        x = int(np.mean(x_points))
        y = int(np.mean(y_points))
        # avoid to beyond boundaries of array
        x = max(56, min(x, 167))
        y = max(56, min(y, 167))
        # get cropped video
        return video[:, x - 56:x + 56, y - 56:y + 56, :]

    def color_jitter(self, video):
        # range of s-component: 0-1
        # range of v component: 0-255
        s_jitter = np.random.uniform(-0.2, 0.2)
        v_jitter = np.random.uniform(-30, 30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[..., 1] + s_jitter
            v = hsv[..., 2] + v_jitter
            s[s < 0] = 0
            s[s > 1] = 1
            v[v < 0] = 0
            v[v > 255] = 255
            hsv[..., 1] = s
            hsv[..., 2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video

    def load_data(self, path):
        # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
        data = np.load(path, mmap_mode='r', allow_pickle=True)
        print("path----------------------------: ", path)
        data = np.float32(data)
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
        # whether to utilize the data augmentation
        if self.data_aug:
            data[..., :3] = self.color_jitter(data[..., :3])
            data = self.random_flip(data, prob=0.5)
        # normalize rgb images and optical flows, respectively
        data[..., :3] = self.normalize(data[..., :3])
        data[..., 3:] = self.normalize(data[..., 3:])
        return data