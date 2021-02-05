# coding=utf-8
import multiprocessing
import os,tqdm, cv2
import threading
import argparse
import torch
import sys
import time
import json

from model import rwf2000_baseline_rgbonly, rwf2000_baseline_flowgate
from util.log import Logger
from util.utils import *
from util.alert_stream import LoadStreams, PushStreams
from util.sampling import *
from util.event import UDPSocket
import uuid

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


def push_stream_process(pre, video_clip_np, putStream, video_clip_np_name, cam_ip):
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
    cur_frame = 0
    VIDEO_PATH_OUT = os.path.dirname(os.path.abspath(__file__)) + "/alert/"+cam_ip+"/video/" + video_clip_np_name + ".avi"
    IMAGE_PATH_OUT = os.path.dirname(os.path.abspath(__file__)) + "/alert/"+cam_ip+"/image/" + video_clip_np_name + ".jpg"
    writer = None
    if pre == 0:
        writer = cv2.VideoWriter(VIDEO_PATH_OUT,
                                 cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                                 25,  # fps
                                 (width, height))  # resolution
        cv2.imwrite(IMAGE_PATH_OUT, video_clip_np[0])         

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
        # print("pushing cur_frame: ", cur_frame, " to ", putStream.command[-1])
        cur_frame += 1


def VideoRealtimeDetectionwithcam(model=None, npy_save_dir = "/home/weida/workspace/Dataset/realdata-npy-noflow/", resize=(224, 224), frame_window=75, frame_interval=64, \
                           target_frames=75, sample='uniform_sampling', gap=2, use_thread=False, loadStreams=None, putStreams=None, udp_socket=None, cam_ip=None):
    # video_path = "/home/weida/workspace/Dataset/realdata/fight_20210128_101800_20210128_103159_1.mp4"
    # cap = cv2.VideoCapture(video_path)
    frames_queue = []
    video_clip = []
    idx = 0
    pre_idx = 0
    # （1）读帧+入队+队列里每满75帧（2）选出前64帧去预测（3）得到结果后，队列里前11帧去推流，紧接队列出列11帧。重复（1）（2）（3）
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
            snippet_data = np.array(frames_queue)#2.用于预测
            video_clip_np = np.array(video_clip[:frame_interval])#3.用于输出11帧
            pred = SnippetViolenceDetection(model, snippet_data, sample, target_frames, gap)
            pred = 0
            pre_idx = pre_idx + frame_interval
            if not os.path.exists(os.path.dirname(os.path.abspath(__file__))+"/alert"):
                os.makedirs(os.path.dirname(os.path.abspath(__file__))+"/alert")
            if  not os.path.exists(os.path.dirname(os.path.abspath(__file__))+"/alert/"+cam_ip):
                os.makedirs(os.path.dirname(os.path.abspath(__file__))+"/alert/"+cam_ip)
            if  not os.path.exists(os.path.dirname(os.path.abspath(__file__))+"/alert/"+cam_ip+"/video"):
                os.makedirs(os.path.dirname(os.path.abspath(__file__))+"/alert/"+cam_ip+"/video")
            if  not os.path.exists(os.path.dirname(os.path.abspath(__file__))+"/alert/"+cam_ip+"/image"):
                os.makedirs(os.path.dirname(os.path.abspath(__file__))+"/alert/"+cam_ip+"/image")
            #对图像进行画框、加字、pushStream等操作-----------------------------------------
            if use_thread == True:
                video_clip_np_name = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
                t = threading.Thread(target=push_stream_process, args=(pred, video_clip_np, putStreams, video_clip_np_name, cam_ip))
                t.start()
                t.join()
            else:
                push_stream_process(pred, video_clip_np, putStreams, video_clip_np_name, cam_ip)
            frames_queue = frames_queue[frame_interval:]#视频序列出队，每次出队步长为 frame_interval
            video_clip = video_clip[frame_interval:]
            data = {
                "type": "alertBegin",
                "time": int(time.time() * 1000),
                "data": {
                    "ip" : cam_ip,
                    "id" : str(uuid.uuid1()),
                    "detail":{# 仅当发生告警是才有这部分数据，其他情况为空
                        "path:" : os.path.dirname(os.path.abspath(__file__))+"/alert/"+cam_ip, 
                        "videoPath" : os.path.dirname(os.path.abspath(__file__))+"/alert/"+cam_ip+"/video/"+video_clip_np_name+".avi",
                        "imagePath" : os.path.dirname(os.path.abspath(__file__))+"/alert/"+cam_ip+"/image/"+video_clip_np_name+".jpg", 
                        "alertType" : 7,
                    }
                }
            }
            # print(os.path.dirname(os.path.abspath(__file__))+"/alert/images/"+video_clip_np_name+".jpg")
            if pred == 0:
                udp_socket.send(json.dumps(data), log=False)

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
    parser.add_argument('-sample', type=str, default='uniform_sampling',
                        choices=['uniform_sampling', 'random_continuous_sampling', 'random_gap_sampling',
                                 'no_sampling'],
                        help='sampling method for sampling snippet frames to model input')
    parser.add_argument('-frame_window', type=int, default=150, help='snippet window length')
    parser.add_argument('-frame_interval', type=int, default=11, help='sliding interval')
    parser.add_argument('-target_frames', type=int, default=64, help='model input frames')
    parser.add_argument('-use_thread', type=bool, default=True, help='pushStream with thread or not')
    parser.add_argument('-gap', type=int, default=2, help='only for gap sampling')
    #maybe you want to edit below======================
    parser.add_argument('-source_url', type=str, default="rtsp://admin:zhks2020@192.168.1.6:554/Streaming/Channels/1", help='load stream camera source url')
    parser.add_argument('-rtmp_address', type=str, default="192.168.1.201:1936", help='push stream address')
    parser.add_argument('-cam_index', type=int, default="2", help='push stream cam num')
    parser.add_argument('-socket_address', type=str, default="192.168.1.199:6666", help='socke_address')
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

    socket_address = args.socket_address
    host, port = socket_address.split(':')
    port = int(port)
    udp_socket = UDPSocket(host=host, port=port)

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
    cam_ip = source_url[source_url.find("@")+1:source_url.rfind(":554/Streaming")]
    # print("cam_ip: ", cam_ip)
    source_url = args.source_url
    loadStreams = LoadStreams(source_url, img_size=(224, 224))
    # rtmp_address = "192.168.1.23:1936"
    rtmp_address = args.rtmp_address
    cam_index = args.cam_index
    putStreams = PushStreams(index=cam_index, rtmp_address=rtmp_address, test=0)

    VideoRealtimeDetectionwithcam(model = model, resize=(224, 224), frame_window=frame_window, frame_interval=frame_interval, \
                                  target_frames=target_frames, gap = gap, use_thread = use_thread, loadStreams = loadStreams, putStreams = putStreams, \
                                  udp_socket = udp_socket, cam_ip=cam_ip)