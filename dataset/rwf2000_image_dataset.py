import os, torch, random, cv2, glob
import numpy as np
from torch.utils import data

class rwfdataset(data.Dataset):
	def __init__(self, dataset_dir, size, seq_len, transform, gap=1, mode='train'):
		self.size = size
		self.seq_len = seq_len
		self.dataset_dir = dataset_dir
		self.transform = transform
		self.gap = gap
		self.mode = mode

		self.X_path, self.Y_dict = self.search_data()
		# Print basic statistics information
		self.print_stats()

	def search_data(self):
		X_path = []
		Y_dict = {}
		# list all kinds of sub-folders
		self.dirs = sorted(os.listdir(self.dataset_dir))
		labels = range(len(self.dirs))
		for i, folder in enumerate(self.dirs):
			folder_path = os.path.join(self.dataset_dir, folder)
			for instance in os.listdir(folder_path):
				instance_path = os.path.join(folder_path, instance)
				X_path.append(instance_path)
				Y_dict[instance_path] = labels[i]
		return X_path, Y_dict

	def print_stats(self):
		# calculate basic information
		self.n_instances = len(self.X_path)
		self.n_classes = len(self.dirs)
		self.indexes = np.arange(len(self.X_path))
		print("Found {} instances belonging to {} classes.".format(self.n_instances, self.n_classes))
		for i, label in enumerate(self.dirs):
			print('%10s : ' % (label), i)
		return None

	def __getitem__(self, index):
		im_dir = self.X_path[index]
		label = self.Y_dict[im_dir]
		image_list = glob.glob(im_dir + "/*.jpg")
		image_list.sort()
		assert len(image_list) == 150, im_dir+' frame num is not 150!'
		if self.mode == 'train':
			im_paths = []
			rand_frame = int(random.random()*(len(image_list)-(self.seq_len-1)*self.gap))
			for i in range(rand_frame, rand_frame+(self.seq_len-1)*self.gap+1, self.gap):
				im_path = image_list[i]
				im_paths.append(im_path)
		elif self.mode == 'val':
				im_paths = image_list

		frames = []
		for im_path in im_paths:
			image = cv2.imread(im_path)
			image = cv2.resize(image, self.size)
			image = image[:, :, ::-1]
			image = self.transform(image.copy())
			frames.append(image.numpy())

		frames = np.array(frames, np.float32)
		frames = torch.from_numpy(frames).float()

		return frames, label

	def __len__(self):
		return len(self.X_path)