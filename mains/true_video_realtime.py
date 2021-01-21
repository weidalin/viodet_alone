# coding=utf-8
import multiprocessing
import os,tqdm, cv2
import threading
import random
from queue import Queue

import numpy as np
import torch
import sys
import time

import subprocess

from alert_stream import LoadStreams

sys.path.append('../')
from util.log import Logger
from model import rwf2000_baseline_rgbonly, rwf2000_baseline_flowgate
from time import sleep


class PushStreams(object):
    """push streams to front website with rtmp
    """
    def __init__(self, index: int, rtmp_address: str, fps=25,
                 resolution=(1920, 1080), test=0) -> None:
        """Initialize
        Args:
            camera_count (int): Camera amount, which equate to push target count
            rtmp_base (str): rtmp address, like 192.168.1.23:1935
            test (int, optional): 0 delegate not test. 1 means test by cv2 show video,
                2 represent test by save video. Defaults to 0.
        """
        super().__init__()
        # Avoid read incorrect fps from source
        if fps <= 5 or fps >= 100:
            fps = 25
        fps = str(fps)
        self.rtmp_base = f"rtmp://{rtmp_address}/myapp/cam"
        self.command = [
            'ffmpeg', '-hwaccel_output_format', 'cuda',  # 使用cuda编码推流
            '-hwaccel', 'cuvid', '-hwaccel_device', '2', '-re', '-f', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s',
            "{}x{}".format(resolution[0], resolution[1]),  # 图片分辨率
            '-r', fps,  # 视频帧率
            '-i', '-', '-c:v', 'h264_nvenc', '-pix_fmt', 'yuv420p',
            '-f', 'flv', '-vsync', '0',
            None  # 接受rtmp的地址
        ]
        self.count = 0
        self.test = test
        # XXX(duke) for video save test'
        # if self.test == 2:
        #     self.video_save = VideoSave('./', 0)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        self.command[-1] = self.rtmp_base + str(index)
        # self.output = subprocess.Popen(self.command, stdin=subprocess.PIPE)
        # NOTE(duke) 下面这个加了重定向的代码使得延迟翻倍
        self.output = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                                       #stdout=subprocess.DEVNULL,
                                       #stderr=subprocess.DEVNULL,
                                       env=env
                                    )
    def push(self, img):
        """push video stream to front end
        Args:
            imgs (list): list of images, consist of numpy.ndarray or torch.tensor
        """
        if isinstance(img, list):
            for i, img_ in enumerate(img):
                if self.test == 1:
                    cv2.imshow('camera_' + str(i), img_)
                elif self.test == 2:
                    self.video_save.add(img_)
                else:
                    self.output.stdin.write(img_.tostring())
        else:
            if self.test == 1:
                cv2.imshow('camera' , img)
                cv2.waitKey(10)
            elif self.test == 2:
                self.video_save.add(img)
            else:
                self.output.stdin.write(img.tostring())

        # print("putstream to :", self.command[-1],"-------------------------------------------------------")


def getOpticalFlow(video):
    """Calculate dense optical flow of input video
    Args:
        video: the input video with shape of [frames,height,width,channel]. dtype=np.array
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
        flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    """
    # initialize the list of optical flows
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img, (224, 224, 1)))

    flows = []
    for i in range(0, len(video) - 1):
        # calculate optical flow between each pair of frames
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        # Add into list
        flows.append(flow)

    # Padding the last frame as empty array
    flows.append(np.zeros((224, 224, 2)))

    return np.array(flows, dtype=np.float32)

def Frames2Npy(frames, save_path=None):
    """Load video and tansfer it into .npy format
    Args:
        file_path: the path of video file
        resize: the target resolution of output video
    Returns:
        frames: gray-scale video
        flows: magnitude video of optical flows
    """
    # Get the optical flow of video
    flows = getOpticalFlow(frames)

    result = np.zeros((len(flows), 224, 224, 5))

    result[..., :3] = frames
    result[..., 3:] = flows

    if save_path is not None:
        np.save(save_path, result)
    return result

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

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def SnippetViolenceDetection(model, data, sample, target_frames, gap):
    #sample
    if sample == 'uniform_sampling':
        data = uniform_sampling(video=data, target_frames=target_frames)
    elif sample == 'random_continuous_sampling':
        data = random_continuous_sampling(video=data, target_frames=target_frames)
    elif sample == 'random_gap_sampling':
        data = random_gap_sampling(video=data, target_frames=target_frames, gap=gap)
    elif sample == 'no_sampling':
        data = data
    # normalize rgb images and optical flows, respectively
    data[..., :3] = normalize(data[..., :3])
    data[..., 3:] = normalize(data[..., 3:])
    data = torch.from_numpy(data).float()

    data = data.unsqueeze(dim=0)
    data = data.cuda()
    data = data[..., :3]
    data = data.permute(0, -1, 1, 2, 3)
    out = model(data)

    _, pred = torch.max(out, 1)
    return pred[0].item()


def push_stream_process(pre, video_clip_np, putStream):
    color = {'green': (0, 255, 0),
             'blue': (255, 165, 0),
             'dark red': (0, 0, 139),
             'red': (0, 0, 255),
             'dark slate blue': (139, 61, 72),
             'aqua': (255, 255, 0),
             'brown': (42, 42, 165),
             'deep pink': (147, 20, 255),
             'fuchisia': (255, 0, 255),
             'yello': (0, 238, 238),
             'orange': (0, 165, 255),
             'saddle brown': (19, 69, 139),
             'black': (0, 0, 0),
             'white': (255, 255, 255)}
    # print("video_clip_np length: ", len(video_clip_np))
    width = video_clip_np.shape[2]
    height = video_clip_np.shape[1]
    cur_frame = 0;
    VIDEO_PATH_OUT = "/home/weida/workspace/Dataset/violence_clip_backup/violence_clip_" + time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time())) + ".avi"
    writer = None
    if pre == 0:
        writer = cv2.VideoWriter(VIDEO_PATH_OUT,
                                 cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                                 25,  # fps
                                 (width, height))  # resolution

    while cur_frame < video_clip_np.shape[0]:
        # print("cur_frame: ", cur_frame)
        VIOLENCE_TAG = "NO"
        #如果是暴力
        if pre == 0:
            VIOLENCE_TAG = "YES"
            cv2.rectangle(video_clip_np[cur_frame], (0, 0), (width - 1, height - 1), color["red"], 10)
            cv2.putText(video_clip_np[cur_frame], "violence:" + VIOLENCE_TAG, (50, 30 - 7), cv2.FONT_ITALIC, 1, color["red"], 2)
            writer.write(video_clip_np[cur_frame])
        #不是暴力
        else:
            cv2.putText(video_clip_np[cur_frame], "violence:" + VIOLENCE_TAG, (50, 30 - 7), cv2.FONT_ITALIC, 1,color["green"], 2)
        putStream.push(video_clip_np[cur_frame])
        print("pushing cur_frame: ", cur_frame, " to ", putStream.command[-1])
        cur_frame += 1

    # cv2.destroyAllWindows()

def VideoRealtimeDetectionwithcam(model, npy_save_dir=None, resize=(224, 224), frame_window=150, frame_interval=75,
                           target_frames=64, sample='uniform_sampling', gap=2):
    # Load video
    source_url = "rtsp://admin:scut123456@192.168.1.65:554/Streaming/Channels/1"
    loadStreams = LoadStreams(source_url, img_size=(224, 224))

    frames_queue = []
    video_clip = []
    idx = 0
    pre_idx = 0
    res = []
    cur_frame = 0
    putStream = PushStreams(index=0, rtmp_address="192.168.1.23:1936")
    while True:
        idx += 1
        source, frame, img_raw, timestamp = loadStreams.__next__()
        if frame is None:
            continue
        video_clip.append(img_raw)

        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (224, 224, 3))
        frames_queue.append(frame)
        if len(frames_queue)==frame_interval:
            #推流原始数据
            video_clip_np = np.array(video_clip[:target_frames])
            snippet_data = np.array(frames_queue[:target_frames])
            npy_save_path = None
            if npy_save_dir is not None:
                if not os.path.exists(npy_save_dir): os.makedirs(npy_save_dir)
                # npy_save_path = os.path.join(npy_save_dir, str(pre_idx)+'-'+str(idx))
                npy_save_path = os.path.join(npy_save_dir, "tmp")
            snippet_data = Frames2Npy(snippet_data, npy_save_path)
            pred = SnippetViolenceDetection(model, snippet_data, sample, target_frames, gap)
            res.append(str(pre_idx) + '-' + str(idx) + ',' + str(pred))
            print("idx - pre_idx:", idx, "-", pre_idx, "=", idx - pre_idx)
            pre_idx = pre_idx + frame_interval
            frames_queue = frames_queue[frame_interval:]

            #对图像进行画框、加字、pushStream等操作-------------------------------------
            # t1 = threading.Thread(target = push_stream_process,  args=(pred, video_clip_np, putStream ))
            t = threading.Thread(target=push_stream_process, args=(pred, video_clip_np, putStream ))
            t.start()
            t.join()
            # push_stream_process(pred, video_clip_np, putStream )
            video_clip = video_clip[frame_interval:]



import argparse
def true_video_realtime_main():
    print("from cam_detect_realtime......")
    parser = argparse.ArgumentParser('Real Date video violence detect by detecting npys.')
    parser.add_argument('-gpu', type=str, default='0', help='gpu ids to use')
    parser.add_argument('-model_path', type=str, default='weight/rwf_baseline_noflow', help='model weights dir')
    parser.add_argument('-checkpoint', type=str, default='epoch25_0.8475.pth', help='checkpoint filename.')
    parser.add_argument('-model_type', type=str, default='rgbonly', choices=['rgbonly', 'flowgate'])
    parser.add_argument('-logger_path', type=str, default=None, help='logger file')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch_size of test_dataloader')
    parser.add_argument('-j', '--num_workers', type=int, default=16, help='worker number.')
    parser.add_argument('-true_video_dir', type=str, default='/home/weida/workspace/Dataset/violence2/',
                        help='Real data .npy files dir')
    parser.add_argument('-npy_save_path', type=str, default="/home/weida/workspace/Dataset/realdata-npy-noflow",
                        help='dir to save intermediate npy data')
    parser.add_argument('-frame_window', type=int, default=150, help='snippet window length')
    parser.add_argument('-frame_interval', type=int, default=75, help='sliding interval')
    parser.add_argument('-sample', type=str, default='uniform_sampling',
                        choices=['uniform_sampling', 'random_continuous_sampling', 'random_gap_sampling',
                                 'no_sampling'],
                        help='sampling method for sampling snippet frames to model input')
    parser.add_argument('-target_frames', type=int, default=64, help='model input frames')
    parser.add_argument('-gap', type=int, default=2, help='only for gap sampling')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_path = args.model_path
    checkpoint = args.checkpoint
    model_type = args.model_type
    print('model_path: ', model_path + '/' + checkpoint)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if args.logger_path is None:
        sys.stdout = Logger(model_path + '/log_realtime.txt')
    else:
        sys.stdout = Logger(model_path + '/log_realtime.txt')

    if model_type == 'rgbonly':
        model = rwf2000_baseline_rgbonly.RWF_RGB()
    elif model_type == 'flowgate':
        model = rwf2000_baseline_flowgate.RWF_FlowGate()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join(model_path, checkpoint)))

    model.eval()

    true_video_dir = args.true_video_dir
    frame_window = args.frame_window
    #frame_interval = args.
    frame_interval = 64
    #target_frames = args.target_frames
    target_frames = 64
    sample = args.sample
    gap = args.gap
    npy_save_path = args.npy_save_path
    if npy_save_path is None:
        npy_save_path = "/home/weida/workspace/Dataset/realdata-npy-noflow_" + str(frame_window) + "_" + str(
            frame_interval)
    npy_save_dir = "/home/weida/workspace/Dataset/realdata-npy-noflow/"
    VideoRealtimeDetectionwithcam(model,  npy_save_dir=npy_save_dir,
                                  frame_window=target_frames)

    # videos = os.listdir(true_video_dir)
    # for v in videos:
    #     # if 'fight' in v:
    #     npy_save_dir = os.path.join(npy_save_path, v[:-4])
    #     video_path = os.path.join(true_video_dir, v)
    #
    #
    #
    #     results, video_frame_res = VideoRealtimeDetection(model, file_path=video_path, npy_save_dir=npy_save_dir,
    #                                      frame_window=args.target_frames)
    #     f = open(model_path + '/true_video_realtime_' + v[:-4] + '.txt', 'w')
    #     for res in results:
    #         f.write(res + '\n')
    #     f.close()
    # return video_frame_res, video_path

if __name__ == '__main__':
    true_video_realtime_main()
