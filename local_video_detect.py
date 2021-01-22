# coding=utf-8
import multiprocessing
import os,tqdm, cv2
import threading
import argparse
import torch
import sys
import time

from model import rwf2000_baseline_rgbonly, rwf2000_baseline_flowgate
from util.log import Logger
from util.utils import *
from util.alert_stream import LoadStreams, PushStreams
from util.sampling import *


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
    data[..., :3] = normalize(data[..., :3]) #rgb
    data[..., 3:] = normalize(data[..., 3:]) #flow
    data = torch.from_numpy(data).float()

    data = data.unsqueeze(dim=0)
    data = data.cuda()
    data = data[..., :3]
    data = data.permute(0, -1, 1, 2, 3)
    out = model(data)

    _, pred = torch.max(out, 1)
    return pred[0].item()


def show_video_clip(pre, video_clip_np):
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
        cv2.imshow("cur_frame", video_clip_np[cur_frame])
        cv2.waitKey(10)
        cur_frame += 1


def local_video_detect(model, video_dir = None, npy_save_dir = "/home/weida/workspace/Dataset/realdata-npy-noflow/", resize=(224, 224), frame_window=150, frame_interval=64,
                           target_frames=75, sample='uniform_sampling', gap=2, use_thread=False, loadStreams=None, putStreams=None):
    if os.path.isfile(video_dir):
        videos = [video_dir]
    else:
         videos = [video_dir+video for video in os.listdir(video_dir)]

    # 1.读帧+入队+队列里每满150帧，2.选出前75帧作为infer，3前65帧去推流，紧接队列出列64帧,重复123
    for VIDEO_PATH in videos:
        cap = cv2.VideoCapture(VIDEO_PATH)
        len_frames = int(cap.get(7))
        if cap.isOpened() == False:
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_queue = []
        video_clip = []
        idx = 0
        pre_idx = 0
        res = []
        VIDEO_PATH_OUT = "/home/weida/workspace/Dataset/violence_clip_backup/"+ time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))+".avi"
        writer = None
        # writer = cv2.VideoWriter(VIDEO_PATH_OUT,
        #                          cv2.VideoWriter_fourcc('I', '4', '2', '0'),
        #                          25,  # fps
        #                          (width, height))  # resolution

        video_interval_begin = time.time()
        time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        while (cap.isOpened()) and idx < len_frames:
            cur_frame = 0
            have_more_frame, frame = cap.read()  # 将视频帧读取到cv::Mat矩阵中
            video_clip.append(frame)
            idx += 1
            if frame is None:
                continue
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224, 224, 3))
            frames_queue.append(frame)
            if len(frames_queue) == frame_window:
                snippet_data = np.array(frames_queue)
                npy_save_path = None
                if npy_save_dir is not None:
                    if not os.path.exists(npy_save_dir): os.makedirs(npy_save_dir)
                    # npy_save_path = os.path.join(npy_save_dir, str(pre_idx) + '-' + str(idx))
                    npy_save_path = os.path.join(npy_save_dir,"tmp")
                snippet_data = Frames2Npy(snippet_data, npy_save_path)
                pred = SnippetViolenceDetection(model, snippet_data, sample, target_frames, gap)
                res.append(str(pre_idx) + '-' + str(idx) + ', pred: ' + str(pred))
                print("idx - pre_idx:", idx, "-", pre_idx, "=", idx - pre_idx,' pred: ' + str(pred))
                pre_idx = pre_idx + frame_interval
                if use_thread:
                    None
                    t = threading.Thread(target=show_video_clip, args=(pred, np.array(video_clip[ : frame_interval])))
                    t.start()
                    t.join()
                else:
                    show_video_clip(pred, np.array(video_clip[ : frame_interval]))
                frames_queue = frames_queue[frame_interval : ]
                video_clip = video_clip[frame_interval:]
        video_interval_end = time.time()
        print("this video cost time :", video_interval_end - video_interval_begin)
        cap.release()
        # return res, video_frame_res,


if __name__ == '__main__':
    print("from cam_detect_realtime......")
    parser = argparse.ArgumentParser('Real Date video violence detect by detecting npys.')
    parser.add_argument('-gpu', type=str, default='0', help='gpu ids to use')
    parser.add_argument('-model_path', type=str, default='mains/weight/rwf_baseline_noflow', help='model weights dir')
    parser.add_argument('-checkpoint', type=str, default='epoch25_0.8475.pth', help='checkpoint filename.')
    parser.add_argument('-model_type', type=str, default='rgbonly', choices=['rgbonly', 'flowgate'])
    parser.add_argument('-logger_path', type=str, default=None, help='logger file')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch_size of test_dataloader')
    parser.add_argument('-j', '--num_workers', type=int, default=16, help='worker number.')
    parser.add_argument('-video_dir', type=str, default='/home/weida/workspace/Dataset/violence2/',
                        help='Real data .npy files dir')
    parser.add_argument('-npy_save_path', type=str, default="/home/weida/workspace/Dataset/realdata-npy-noflow",
                        help='dir to save intermediate npy data')
    parser.add_argument('-sample', type=str, default='uniform_sampling',
                        choices=['uniform_sampling', 'random_continuous_sampling', 'random_gap_sampling',
                                 'no_sampling'],
                        help='sampling method for sampling snippet frames to model input')
    parser.add_argument('-frame_window', type=int, default=75, help='snippet window length')
    parser.add_argument('-frame_interval', type=int, default=10, help='sliding interval')
    parser.add_argument('-target_frames', type=int, default=64, help='model input frames')
    parser.add_argument('-gap', type=int, default=2, help='only for gap sampling')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_path = args.model_path
    video_dir = args.video_dir
    checkpoint = args.checkpoint
    model_type = args.model_type
    frame_window = args.frame_window
    target_frames = args.target_frames
    frame_interval = args.frame_interval
    sample = args.sample
    gap = args.gap
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
    # Load video
    # source_url = "rtsp://admin:scut123456@192.168.1.65:554/Streaming/Channels/1"
    # loadStreams = LoadStreams(source_url, img_size=(224, 224))
    # rtmp_address = "192.168.1.23:1936"
    # putStreams = PushStreams(index=6, rtmp_address=rtmp_address, test=1)

    local_video_detect(model = model, video_dir = video_dir, resize=(224, 224), frame_window=frame_window, frame_interval=frame_interval,
                                  target_frames=target_frames, gap = gap, use_thread=True, loadStreams = None, putStreams = None)