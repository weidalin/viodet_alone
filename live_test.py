# coding=utf-8
import queue
import threading
import cv2 as cv
import subprocess as sp


class Live(object):
    def __init__(self):
        self.frame_queue = queue.Queue()
        self.command = ""
        self.rtspUrl = "rtsp://localhost:8554/mystream3"
        self.camera_path = "rtsp://admin:jiayuan123@192.168.1.65:554/Streaming/Channels/1"

    def read_frame(self):
        print("开启推流")
        cap = cv.VideoCapture(self.camera_path)

        # Get video information
        fps = int(cap.get(cv.CAP_PROP_FPS))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # ffmpeg command
        self.command = ['ffmpeg',
                        '-y',
                        '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', "{}x{}".format(width, height),
                        '-r', str(fps),
                        '-i', '-',
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', 'ultrafast',
                        '-f', 'rtsp',
                        self.rtspUrl]

        # read webcamera
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print("Opening camera is failed")
                break

            # put frame into queue
            self.frame_queue.put(frame)

    def push_frame(self):
        # 防止多线程时 command 未被设置
        while True:
            if len(self.command) > 0:
                print(self.command)
                # 管道配置
                p = sp.Popen(self.command, stdin=sp.PIPE)
                break

        while True:
            if self.frame_queue.empty() != True:
                frame = self.frame_queue.get()
                # process frame
                # 你处理图片的代码
                # write to pipe
                p.stdin.write(frame.tostring())

    def run(self):
        threads = [
            threading.Thread(target=Live.read_frame, args=(self,)),
            threading.Thread(target=Live.push_frame, args=(self,))
        ]
        [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]

# if __name__ == '__main__':
#     live = Live()
#     live.run()

# import subprocess as sp
#
# rtspUrl = "rtsp://localhost:8554/mystream3"
# camera_path = "rtsp://admin:jiayuan123@192.168.1.65:554/Streaming/Channels/1"
# cap = cv.VideoCapture(camera_path)
#
# # Get video information
# fps = int(cap.get(cv.CAP_PROP_FPS))
# width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#
# # ffmpeg command
# command = ['ffmpeg',
#            '-y',
#            '-f', 'rawvideo',
#            '-vcodec', 'rawvideo',
#            '-pix_fmt', 'bgr24',
#            '-s', "{}x{}".format(width, height),
#            '-r', str(fps),
#            '-i', '-',
#            '-c:v', 'libx264',
#            '-pix_fmt', 'yuv420p',
#            '-preset', 'ultrafast',
#            '-f', 'rtsp',
#            rtspUrl]
#
# # 管道配置
# p = sp.Popen(command, stdin=sp.PIPE, shell=True)
#
# # read webcamera
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if not ret:
#         print("Opening camera is failed")
#         break
#
#     # process frame
#     # your code
#     # process frame
#
#     # write to pipe
#     p.stdin.write(frame.tostring())

import cv2
import queue
import os
import numpy as np
from threading import Thread
import datetime, _thread
import subprocess as sp
import time

# 使用线程锁，防止线程死锁
mutex = _thread.allocate_lock()
# 存图片的队列
frame_queue = queue.Queue()
rtmp_address = "192.168.1.23:1936"
rtspUrl = "rtmp://%s/myapp/cam3"%rtmp_address
camera_path = "rtsp://admin:scut123456@192.168.1.65:554/Streaming/Channels/1"

# Get video information
cap = cv.VideoCapture(camera_path)
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# print(width, height, fps)
cap.release()

# 用于推流的配置,参数比较多，可网上查询理解
command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(width, height),  # 图片分辨率
           '-r', str(fps),  # 视频帧率
           '-i', '-',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'rtsp',
           rtspUrl]


def Video():
    # 调用相机拍图的函数
    vid = cv2.VideoCapture(camera_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    while (vid.isOpened()):
        return_value, frame = vid.read()

        # 原始图片推入队列中
        frame_queue.put(frame)


def push_frame():
    # 推流函数

    # 防止多线程时 command 未被设置
    while True:
        if len(command) > 0:
            # 管道配置，其中用到管道
            p = sp.Popen(command, stdin=sp.PIPE)
            print(command)
            break

    while True:
        if frame_queue.empty() != True:
            # 从队列中取出图片
            frame = frame_queue.get()

            # process frame
            # 你处理图片的代码
            # 将图片从队列中取出来做处理，然后再通过管道推送到服务器上

            # 增加画面帧率
            # if accum_time > 1:
            # accum_time = accum_time - 1
            # fps = "FPS: " + str(curr_fps)
            # curr_fps = 0

            # write to pipe
            # 将处理后的图片通过管道推送到服务器上,image是处理后的图片
            if frame is None:
                continue
            p.stdin.write(frame.tostring())


def run():
    # 使用两个线程处理

    thread1 = Thread(target=Video, )
    thread1.start()
    thread2 = Thread(target=push_frame, )
    thread2.start()


if __name__ == '__main__':
    run()