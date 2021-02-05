from queue import Queue
import queue
import threading
from typing import Deque
import cv2
import logging
import socket
import time
import os
import json
import multiprocessing
import numpy as np
import ctypes
import subprocess


class Event(object):
    """ Event that will started by monitor during some special situation

    约定：
        这个事件必须是线程安全的，当一个事件发生时，monitor会去实例化一个Event类，
        并通过协程来运行。
    """
    def __init__(self) -> None:
        super().__init__()

    def close(self):
        pass


class SingletonMetaClass(type):
 
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__call__(*args, **kwargs)  # __call__ here is __init__ of new object
        return cls._instance
 

class UDPSocket(object, metaclass=SingletonMetaClass):
    """Send a udp socket message to backend Springboot server

    约定:
        在实例化时创建唯一的相关socket连接，然后每次send数据，系统关闭时关闭连接。
    """

    def __init__(self, host='localhost', port=6666) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data: str, log=True):
        if log:
            logging.info(f'send data: {data}') 
        self.socket.sendto(data.encode('utf-8'), (self.host, self.port))

    def close(self):
        self.socket.close()
        logging.warning(f'Connection with backend server lost')


class AlertSend(object):
    """Using to send an alert

    约定：
        将发送警报划分为3个阶段，开始和持续需要外部调用，自己控制结束。（或者开始和结束外部调用，
        自己控制持续）
    """
    def __init__(self, socket: UDPSocket, alert_id: str, save_path: str, 
                 ip: str = None, alert_type=1) -> None:
        super().__init__()
        self.socket = socket
        self.id = alert_id
        self.live = True
        self.save_path = save_path
        self.ip = ip
        self.alert_type = alert_type

    def begin(self, mtype='alertBegin', other=None):
        d = {
            "type" : mtype,
            "time" : int(time.time() * 1000),  # long类型  作为告警发生时间happenTime
            "data" : {
                "ip" : self.ip,  # 摄像头的IP地址
                "id" : self.id,  # uuid，唯一标识，标识该事件
                "detail": {  # 仅当发生告警是才有这部分数据，其他情况为空
                    "path" : self.save_path,  # 告警视频存放路径, 告警视频的第一帧作为封面
                    "alertType" : self.alert_type,  # 告警类型编号
                }
            }
        }   
        if other:
            d['data']['detail']['other'] = other
        self.socket.send(json.dumps(d))

    def sustain(self):
        # d = {
        #     "type" : "alertBegin",
        #     "time" : int(time.time() * 1000),  # long类型  作为告警发生时间happenTime
        #     "data" : {
        #         "ip" : "192.168.1.12", # 摄像头的IP地址
        #         "id" : self.id,  # uuid，唯一标识，标识该事件
        #         "detail": { # 仅当发生告警是才有这部分数据，其他情况为空
        #             "path" : self.save_path, # 告警视频存放路径, 告警视频的第一帧作为封面
        #             "alertType" : 1, # 告警类型编号
        #         }
        #     }
        # }   
        # self.socket.send(json.dumps(d))
        pass

    def close(self):
        # d = {
        #     "type" : "alertEnd",
        #     "time" : int(time.time() * 1000),  # long类型  作为告警发生时间happenTime
        #     "data" : {
        #             "id" : self.id,  # uuid，唯一标识，标识该事件
        #         }
        # }
        # self.socket.send(json.dumps(d))
        pass


class VideoWriter(Event):
    """Write a video to disk with opencv

    约定:
        该类别会在实例化时创建opencv的writer，然后尽可能高性能地将图像写入磁盘（可能是
        一次性写入许多图像，也可能每次输入一张图像。类结束时，需释放writter
    """
    def __init__(self, path, fps=25, resolution=(1920, 1080), backend='ffmpeg') -> None:
        """Initialize opencv video writer with some parameters.

        Args:
            path (str): directory to save video
            id (str): event id
        """
        super().__init__()
        self.path = path
        self.backend = backend

        if backend == 'opencv':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(path, fourcc, fps, resolution)
        elif backend == 'ffmpeg':
            command = [
                'ffmpeg',
                '-hwaccel_output_format', 'cuda',
                '-hwaccel', 'cuvid',
                '-hwaccel_device', '0', 
                '-f', 'rawvideo', 
                '-re',
                '-pix_fmt', 'bgr24', 
                '-s', "{}x{}".format(resolution[0], resolution[1]),    
                '-r', str(fps),
                '-i', '-', 
                '-c:v', 'h264_nvenc', 
                '-pix_fmt', 'yuv420p', 
                '-f', 'mp4',
                path]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '2'
            self.writer = subprocess.Popen(command, stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL, 
                                           stderr=subprocess.DEVNULL,
                                           env=env)
        self.home = True

    def write(self, img) -> None:
        try:
            if self.backend == 'opencv':
                self.writer.write(img)
            elif self.backend == 'ffmpeg':
                self.writer.stdin.write(img.tostring())
        except:
            # self.close()
            raise IOError('corrupt while writing videos')

    def close(self):
        if self.backend == 'opencv':
            self.writer.release()
        elif self.backend == 'ffmpeg':
            self.writer.terminate()


class VideoSave(object):
    """Write images into video, and save in disk.
    """
    def __init__(self, path: str, alert_id, fps=25, shape=(1080, 1920, 3)) -> None:
        """init video writer and start a threading

        Args:
            path (str): Path to save video.
        """
        super().__init__()
        self.video_path = os.path.join(path, 'video', str(alert_id)+'.mp4')
        self.img_path = os.path.join(path, 'image', str(alert_id)+'.jpg')
        self.shape = shape
        self.data = queue.Queue()
        self.live = True
        self.fps = fps

        t = threading.Thread(
            target=self.read_imgs,
            daemon=True
        )
        t.start()
        logging.debug(f'Thread [{threading.current_thread().name}] begin video save')

    def sustain(self, data):
        if isinstance(data, Deque):
            while len(data) > 0:
                d = data.popleft()
                self.data.put(d)
        else:
            self.data.put(data)

    def read_imgs(self):
        init_arr = np.zeros(self.shape, dtype=np.uint8)
        arr_shared = multiprocessing.RawArray(ctypes.c_uint8, init_arr.ravel())
        read_event = multiprocessing.Event()
        write_event = multiprocessing.Event()
        read_event.set()
        write_event.clear()
        end_sema = multiprocessing.Semaphore(0)
        shared_space = np.frombuffer(arr_shared, np.uint8).reshape(self.shape)
        p = multiprocessing.Process(
            target=write_imgs, 
            args=(arr_shared, read_event, write_event, self.video_path, 
                  self.img_path, self.shape, end_sema, self.fps, (self.shape[1], self.shape[0])),
            daemon=True)
        p.start()
        while self.live is True:
            read_event.wait()
            try:
                d = self.data.get(timeout=0.5)
            except Exception:
                break
            shared_space[:] = d
            read_event.clear()
            write_event.set()
        end_sema.release()

    def close(self):
        self.live = False

def write_imgs(arr_shared, read_event, write_event, video_path, 
               img_path, shape, end_sema, fps, resolution):
    writer = VideoWriter(video_path, fps=fps, resolution=resolution)
    home = True
    logging.basicConfig(level=logging.DEBUG)
    while True:
        write_event.wait(0.5)
        if end_sema.acquire(False):
            break
        img = np.frombuffer(arr_shared, np.uint8).reshape(shape)
        if home is True:
            cv2.imwrite(img_path, img)
            home = False
        writer.write(img)
        write_event.clear()
        read_event.set()
    writer.close()
    logging.debug(f'Thread [{threading.current_thread().name}] end video save')
