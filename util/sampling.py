import numpy as np

def uniform_sampling(video, target_frames=64):
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