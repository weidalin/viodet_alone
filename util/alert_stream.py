# Dataset utils and dataloaders
import glob
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread
import threading
from typing import Tuple, List
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
import tqdm

# from utils.general import xyxy2xywh, xywh2xyxy
# from utils.torch_utils import torch_distributed_zero_first
# from .event import VideoSave
from queue import Queue


# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)
local_thread = threading.local()


class LoadStreams(object):
    """Read camera stream data, support multi-stream and single stream.

    Stream protocol includes rtsp and so on.
    Return source addresses in list,
    list of img in shape(height, width, channel),
    and raw img in shape (batch, channel, height, width)
    """

    def __init__(self, source: str, img_size=(224,224)):
        self.img_size = img_size
        self.source = source

        # Under fps 25, cache 0.2s frames at most,
        # need 5 * num_cam * size_img memory, 5 * 16 * 2M=160M
        self.imgs = Queue(maxsize=5)
        self.timestamp = None
        self.count = 0;
        s = source
        thread = Thread(target=self.__update, args=(s,), daemon=True)
        thread.start()
        print('')  # newline

        n = 1
        init_img = None
        with tqdm.tqdm(total=n) as pbar:
            for i in range(n):
                init_img = self.imgs.get()
                pbar.update(1)

        # s = letterbox(init_img, new_shape=self.img_size)[0].shape  # inference shapes
        # s = cv2.resize(init_img, (224, 224), interpolation=cv2.INTER_LINEAR).shape

    def __update(self, s):
        local_thread.cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
        if not local_thread.cap.isOpened():
            logging.exception('Failed to open %s' % s)
        else:
            print("cv2.VideoCapture open ", s, "success")
        frame = None
        while frame is None:
            _, frame = local_thread.cap.read()  # guarantee first frame
        self.imgs.put(frame)
        self.w = int(local_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(local_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = local_thread.cap.get(cv2.CAP_PROP_FPS) % 100
        self.fps = local_thread.cap.get(cv2.CAP_PROP_FPS) % 100
        print(' success (%gx%g at %.2f FPS).' % (self.w, self.h, self.fps))

        while local_thread.cap.isOpened():
            try:
                _, frame = local_thread.cap.read()
            except:
                frame = None
                raise IOError('Opencv corrupt while decode stream')
            if frame is not None:
                self.imgs.put(frame)  # 阻塞了
                self.timestamp = time.time()

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self) -> Tuple[str, np.ndarray, np.ndarray, float]:
        self.count += 1
        # logging.debug(f'input queue with {self.imgs[0].qsize()} images')
        img_raw = self.imgs.get()  # BGR, w*h*3

        # weida_Letterbox
        # img = weida_letterbox(img_raw, new_shape=self.img_size, auto=True)
        # Stack
        # img = np.stack(img, 0)

        # Convert
        # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        # img = img[..., ::-1].transpose(2, 0, 1)  # RGB, 3*size*size
        img = cv2.resize(img_raw, (640, 320), interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img)
        return self.source, img, img_raw, self.timestamp

    def __len__(self):
        return self.count  # 1E12 frames = 32 streams at 30 FPS for 30 years


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
            "{}x{}".format(int(resolution[0]), int(resolution[1])),  # 图片分辨率
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

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

# def get_hash(files):
#     # Returns a single hash value of a list of files
#     return sum(os.path.getsize(f) for f in files if os.path.isfile(f))
#
#
# def exif_size(img):
#     # Returns exif-corrected PIL size
#     s = img.size  # (width, height)
#     try:
#         rotation = dict(img._getexif().items())[orientation]
#         if rotation == 6:  # rotation 270
#             s = (s[1], s[0])
#         elif rotation == 8:  # rotation 90
#             s = (s[1], s[0])
#     except:
#         pass
#
#     return s
#
#
# class PushStreams(object):
#     """push streams to front website with rtmp
#     """
#     def __init__(self, index: int, rtmp_address: str, fps=25,
#                  resolution=(1920, 1080), test=0) -> None:
#         """Initialize
#
#         Args:
#             camera_count (int): Camera amount, which equate to push target count
#             rtmp_base (str): rtmp address, like 192.168.1.23:1935
#             test (int, optional): 0 delegate not test. 1 means test by cv2 show video,
#                 2 represent test by save video. Defaults to 0.
#         """
#         super().__init__()
#         # Avoid read incorrect fps from source
#         if fps <= 5 or fps >= 100:
#             fps = 25
#         fps = str(fps)
#         # self.rtmp_base = f"rtmp://{rtmp_address}/live/cam"
#         self.rtmp_base = f"rtmp://{rtmp_address}/myapp/cam"
#         self.command = [
#             'ffmpeg', '-hwaccel_output_format', 'cuda',  # 使用cuda编码推流
#             '-hwaccel', 'cuvid', '-hwaccel_device', '2', '-re', '-f', 'rawvideo',
#             '-pix_fmt', 'bgr24', '-s',
#             "{}x{}".format(resolution[0], resolution[1]),  # 图片分辨率
#             '-r', fps,  # 视频帧率
#             '-i', '-', '-c:v', 'h264_nvenc', '-pix_fmt', 'yuv420p',
#             '-f', 'flv', '-vsync', '0',
#             None  # 接受rtmp的地址
#         ]
#         self.count = 0
#         self.test = test
#         # XXX(duke) for video save test'
#         if self.test == 2:
#             self.video_save = VideoSave('./', 0)
#         env = os.environ.copy()
#         env['CUDA_VISIBLE_DEVICES'] = '2'
#         self.command[-1] = self.rtmp_base + str(index)
#
#         # self.output = subprocess.Popen(self.command, stdin=subprocess.PIPE)
#         # NOTE(duke) 下面这个加了重定向的代码使得延迟翻倍
#         self.output = subprocess.Popen(self.command, stdin=subprocess.PIPE,
#                                        stdout=subprocess.DEVNULL,
#                                        stderr=subprocess.DEVNULL,
#                                        env=env
#                                     )
#
#     def push(self, img):
#         """push video stream to front end
#
#         Args:
#             imgs (list): list of images, consist of numpy.ndarray or torch.tensor
#         """
#         if isinstance(img, list):
#             for i, img_ in enumerate(img):
#                 if self.test == 1:
#                     cv2.imshow('camera_' + str(i), img_)
#                 elif self.test == 2:
#                     self.video_save.add(img_)
#                 else:
#                     self.output.stdin.write(img_.tostring())
#         else:
#             if self.test == 1:
#                 cv2.imshow('camera' , img)
#             elif self.test == 2:
#                 self.video_save.add(img)
#             else:
#                 self.output.stdin.write(img.tostring())
#
#
# class StreamReader(object):
#     def __init__(self) -> None:
#         super().__init__()
#         self.w = None
#         self.h= None
#         self.fps = None
#
#
# class StreamReaderCV2(StreamReader):
#     def __init__(self, source) -> None:
#         super().__init__()
#
#
# class StreamReaderFF(StreamReader):
#     pass





def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_perspective
        # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels

def weida_letterbox(img, new_shape=(360, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

    # # Scale ratio (new / old)
    # ratio = (new_shape[0] / shape[0], new_shape[1] / shape[1])
    #
    # # Compute padding
    # # ratio = r, r  # width, height ratios
    # new_unpad = int(round(shape[1] * ratio[1])), int(round(shape[0] * ratio[0]))
    # dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    # elif scaleFill:  # stretch
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    #
    # dw /= 2  # divide padding into 2 sides
    # dh /= 2
    #
    # if shape[::-1] != new_unpad:  # resize
    #     img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color).transpose(1,2,0)  # add border
    return img


def letterbox(img, new_shape=(360, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)

#
# def extract_boxes(path='../coco128/'):  # from utils.datasets import *; extract_boxes('../coco128')
#     # Convert detection dataset into classification dataset, with one directory per class
#
#     path = Path(path)  # images dir
#     shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
#     files = list(path.rglob('*.*'))
#     n = len(files)  # number of files
#     for im_file in tqdm(files, total=n):
#         if im_file.suffix[1:] in img_formats:
#             # image
#             im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
#             h, w = im.shape[:2]
#
#             # labels
#             lb_file = Path(img2label_paths([str(im_file)])[0])
#             if Path(lb_file).exists():
#                 with open(lb_file, 'r') as f:
#                     lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
#
#                 for j, x in enumerate(lb):
#                     c = int(x[0])  # class
#                     f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
#                     if not f.parent.is_dir():
#                         f.parent.mkdir(parents=True)
#
#                     b = x[1:] * [w, h, w, h]  # box
#                     # b[2:] = b[2:].max()  # rectangle to square
#                     b[2:] = b[2:] * 1.2 + 3  # pad
#                     b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
#
#                     b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
#                     b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
#                     assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'
#
#
# def autosplit(path='../coco128', weights=(0.9, 0.1, 0.0)):  # from utils.datasets import *; autosplit('../coco128')
#     """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
#     # Arguments
#         path:       Path to images directory
#         weights:    Train, val, test weights (list)
#     """
#     path = Path(path)  # images dir
#     files = list(path.rglob('*.*'))
#     n = len(files)  # number of files
#     indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split
#     txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
#     [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing
#     for i, img in tqdm(zip(indices, files), total=n):
#         if img.suffix[1:] in img_formats:
#             with open(path / txt[i], 'a') as f:
#                 f.write(str(img) + '\n')  # add image to txt file
