# coding=utf-8
# from keras.utils import np_utils
import torch
import cv2, random, os
#
# one_hot = np_utils.to_categorical(range(10))
# print(one_hot)
#
# a = torch.randn((2,3))
# b = torch.randn((2,3))
# print(a, b, torch.mul(a, b))

# cap = cv2.VideoCapture('D:\lwg\\5f1d300369f620073e6c6112f8c3d3b4.mp4')
#
# num = 0
# print('all num', cap.get(7))
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = cap.get(cv2.CAP_PROP_FPS)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out = cv2.VideoWriter('D:\lwg\\5f1d300369f620073e6c6112f8c3d3b4_.mp4', fourcc, fps, size)
#
# while (cap.isOpened() and num<=cap.get(7)):
#     ret, frame = cap.read()
#     num += 1
#     if frame is None:
#         print(num, 'is None')
#         continue
#     out.write(frame)
#     if num%500==0:
#         print(num, frame.shape)
#     # print(num, frame.shape)
#     # cv2.imshow('image', frame)
#     k = cv2.waitKey(20)
#     if (k & 0xff == ord('q')):
#         break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# import numpy as np
# a = np.randn((2,3,3))
# print(a, a[..., None])

# random.seed(1)
# os.environ['PYTHONHASHSEED'] = str(1)
# np.random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-
# torch.cuda.manual_seed_all(1)
# print(np.random.randint(150 - 64))

# labels = np.load('D:\lwg\DGAM-Weakly-Supervised-Action-Localization\data\ActivityNet12\\train_data\\train_labels.npy')
# print(labels.shape, labels[0])

# a = torch.randn(3)
# print(a,torch.prod(a))
# a=np.array([1,2,3,4])
# b=np.array([1,1,3,3])
# print((a==b).sum())

def plot_demo_video(video_path, save_path, start, end):
    cap = cv2.VideoCapture(video_path)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(save_path, fourcc, fps, size)
    boxes_f = open('box.txt', 'r')
    boxes = boxes_f.readlines()
    box_data = {}
    for box in boxes:
        id, x1y1x2y2 = box.strip().split(":")
        x1y1x2y2 = x1y1x2y2.split(",")
        # map(int, x1y1x2y2)
        box_data[int(id[:3])] = x1y1x2y2
    while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES)<cap.get(7):
        ret, img = cap.read()
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(frame_id)
        if img is None:
            break
        if frame_id in range(start+1, end):
            # cv2.rectangle(img, (0,0), (size[0]-10, size[1]-10), (0,0,255), 10, 8)
            # img = cv2.putText(img, 'FIGHTing!', (930, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            # img = cv2.rectangle(img, (930, 110), (1300, 610), (0, 0, 255), 5, 8)
            # img = cv2.putText(img, 'FIGHTing!', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # img = cv2.rectangle(img, (400, 50), (570, 260), (0, 0, 255), 2, 8)
            b = box_data[frame_id]
            print("box: ", b)
            img = cv2.putText(img, 'FIGHTing!', (int(b[0])-50, int(b[1])-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            img = cv2.rectangle(img, (int(b[0])-50, int(b[1])-50), (int(b[2])+50, int(b[3])+50), (0, 0, 255), 2, 8)
        out.write(img)
    cap.release()
    out.release()
# plot_demo_video('D:\lwg\\violent_skeleton.mp4', 'D:\lwg\demo_fight1_2.mp4', 460, 640)

# cap = cv2.VideoCapture('D:\lwg\\violent_skeleton.mp4')
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# print(cap.get(7), size)
# cap = cv2.VideoCapture('D:\lwg\\fight1_20200910.mp4')
# print(cap.get(7))
# cap = cv2.VideoCapture('D:\lwg\智能-采区变电所前门 打架1.mp4')
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# print(cap.get(7), size)
# cap = cv2.VideoCapture('D:\lwg\暴力打架识别.avi')
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# print(cap.get(7), size)
# cap = cv2.VideoCapture('D:\lwg\demo\demo_fight1.avi')
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# print(cap.get(7), size)
# cap = cv2.VideoCapture('D:\lwg\\fight1.avi')
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# print(cap.get(7), size)

# cap = cv2.VideoCapture('D:\lwg\\fight2_20200910_.mp4')
# print('all num', cap.get(7))
# num = 0
# while (cap.isOpened() and num<=cap.get(7)):
#     ret, frame = cap.read()
#     num += 1
#     if num in range(350,375):
#         cv2.imwrite('figs/f'+str(num)+'.jpg', frame)
# cap.release()
# cv2.destroyAllWindows()
## 450-650
## 375-725

# a='1','2'
# a = [a]
# print(a, len(a))

def plot_demo_video_(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(save_path, fourcc, fps, size)
    while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES)<cap.get(7):
        ret, img = cap.read()
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(frame_id)
        if img is None:
            break
        # for start,end in times:
        #     if frame_id in range(start+1, end):
        cv2.rectangle(img, (0,0), (size[0], size[1]), (0,0,255), 2, 2)
        img = cv2.putText(img, 'FIGHTing!', (10, 10), 0, 0.3, (0, 0, 255), 1)
        out.write(img)
    cap.release()
    out.release()
# times = [[375,725],[825,974],[1050,1200],[1275,1425],[2100,2249],[2625,2774]]
# plot_demo_video_('D:\lwg\\StreetViolent.mp4', 'D:\lwg\demo_StreetViolent.mp4', times)
# plot_demo_video_('D:\lwg\\StreetFighting.mp4', 'D:\lwg\demo_StreetFighting.mp4')

