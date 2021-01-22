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


def VideoRealtimeDetectionwithcam(model, npy_save_dir = "/home/weida/workspace/Dataset/realdata-npy-noflow/", resize=(224, 224), frame_window=75, frame_interval=64,
                           target_frames=75, sample='uniform_sampling', gap=2, use_thread=False, loadStreams=None, putStreams=None):
    frames_queue = []
    video_clip = []
    idx = 0
    pre_idx = 0
    res = []
    # 1.读帧+入队+队列里每满150帧，2.选出前75帧作为infer，3前65帧去推流，紧接队列出列64帧,重复123
    while True:
        idx += 1
        source, frame, img_raw, timestamp = loadStreams.__next__()
        if frame is None:
            continue
        video_clip.append(img_raw)
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (224, 224, 3))
        frames_queue.append(frame)#1.入队
        if len(frames_queue)==frame_window:#1.当窗口达到150帧
            snippet_data = np.array(frames_queue[:target_frames])#2.用于预测75帧
            video_clip_np = np.array(video_clip[:frame_interval])#3.用于输出64帧
            npy_save_path = None
            if npy_save_dir is not None:
                if not os.path.exists(npy_save_dir): os.makedirs(npy_save_dir)
                # npy_save_path = os.path.join(npy_save_dir, str(pre_idx)+'-'+str(idx))
                npy_save_path = os.path.join(npy_save_dir, "tmp")
            #infer
            snippet_data = Frames2Npy(snippet_data, npy_save_path)
            pred = SnippetViolenceDetection(model, snippet_data, sample, target_frames, gap)
            res.append(str(pre_idx) + '-' + str(idx) + ',' + str(pred))
            print("idx - pre_idx:", idx, "-", pre_idx, "=", idx - pre_idx, "pred:", pred)
            pre_idx = pre_idx + frame_interval
            #对图像进行画框、加字、pushStream等操作-----------------------------------------
            if use_thread == True:
                t = threading.Thread(target=push_stream_process, args=(pred, video_clip_np, putStreams))
                t.start()
                t.join()
            else:
                push_stream_process(pred, video_clip_np, putStreams)
            frames_queue = frames_queue[frame_interval:]#视频序列出队，每次出队步长为 frame_interval
            video_clip = video_clip[frame_interval:]

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
    parser.add_argument('-true_video_dir', type=str, default='/home/weida/workspace/Dataset/violence2/',
                        help='Real data .npy files dir')
    parser.add_argument('-npy_save_path', type=str, default="/home/weida/workspace/Dataset/realdata-npy-noflow",
                        help='dir to save intermediate npy data')
    parser.add_argument('-sample', type=str, default='uniform_sampling',
                        choices=['uniform_sampling', 'random_continuous_sampling', 'random_gap_sampling',
                                 'no_sampling'],
                        help='sampling method for sampling snippet frames to model input')
    parser.add_argument('-frame_window', type=int, default=75, help='snippet window length')
    parser.add_argument('-frame_interval', type=int, default=20, help='sliding interval')
    parser.add_argument('-target_frames', type=int, default=64, help='model input frames')
    parser.add_argument('-use_thread', type=bool, default=False, help='pushStream with thread or not')
    parser.add_argument('-gap', type=int, default=2, help='only for gap sampling')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_path = args.model_path
    checkpoint = args.checkpoint
    model_type = args.model_type
    frame_window = args.frame_window
    target_frames = args.target_frames
    frame_interval = args.frame_interval
    sample = args.sample
    use_thread = args.use_thread
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
    source_url = "rtsp://admin:scut123456@192.168.1.65:554/Streaming/Channels/1"
    loadStreams = LoadStreams(source_url, img_size=(224, 224))
    rtmp_address = "192.168.1.23:1936"
    putStreams = PushStreams(index=6, rtmp_address=rtmp_address, test=0)

    VideoRealtimeDetectionwithcam(model = model, resize=(224, 224), frame_window=frame_window, frame_interval=frame_interval,
                                  target_frames=target_frames, gap = gap, use_thread = use_thread, loadStreams = loadStreams, putStreams = putStreams)