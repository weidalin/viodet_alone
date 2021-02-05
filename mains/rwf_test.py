import os, random, torch, cv2, time, sys
sys.path.append('../')
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from dataset import rwf2000_dataset
from model import rwf2000_baseline_rgbonly, rwf2000_baseline_flowgate
from util.log import Logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RWF video violence detect by detecting npys.')
    parser.add_argument('-gpu', type=str, default='1', help='gpu ids to use')
    parser.add_argument('-model_path', type=str, default='weight/rwf_baseline_noflow', help='model weights dir')
    parser.add_argument('-checkpoint', type=str, default='epoch25_0.8475.pth', help='checkpoint filename.')
    parser.add_argument('-model_type', type=str, default='rgbonly', choices=['rgbonly', 'flowgate'])
    parser.add_argument('-data_root', type=str, default='/home/lwg/workspace/Dataset/RWF-2000')
    parser.add_argument('-data_name', type=str, default='RWF-2000-npy-withflow')
    parser.add_argument('-sample', type=str, default='uniform_sampling',
                        choices=['uniform_sampling', 'random_continuous_sampling', 'random_gap_sampling',
                                 'no_sampling'],
                        help='sampling method for sampling snippet frames to model input')
    parser.add_argument('-target_frames', type=int, default=64, help='model input frames')
    parser.add_argument('-gap', type=int, default=2, help='only for gap sampling')
    parser.add_argument('-logger_path', type=str, default=None, help='logger file')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch_size of test_dataloader')
    parser.add_argument('-j', '--num_workers', type=int, default=16, help='worker number.')
    parser.add_argument('-test_dir', type=str, default='/home/lwg/workspace/Dataset/True Data npy/',
                        help='Real data .npy files dir')
    parser.add_argument('-res_path', type=str, default=None, help='violence detect result file')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_path = args.model_path
    checkpoint = args.checkpoint
    model_type = args.model_type
    batch_size = args.batch_size
    num_workers = args.num_workers
    sample_method = args.sample
    seq_len = args.target_frames
    gap = args.gap
    print('model_path: ', model_path + '/' + checkpoint)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if args.logger_path is None:
        sys.stdout = Logger(model_path + '/log_test.txt')
    else:
        sys.stdout = Logger(args.logger_path)

    data_root = args.data_root
    data_type = args.data_name
    val_dir = os.path.join(data_root, data_type, 'val')

    val_dataset = rwf2000_dataset.RWFDataset(directory=val_dir, data_augmentation=False, target_frames=seq_len,
                                             sample=sample_method, gap=gap)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)

    if model_type == 'rgbonly':
        model = rwf2000_baseline_rgbonly.RWF_RGB()
    elif model_type == 'flowgate':
        model = rwf2000_baseline_flowgate.RWF_FlowGate()
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(model_path, checkpoint)))

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    total_correct = 0.0
    all_label_num = 0.0
    model.eval()
    error_ids = []
    with torch.no_grad():
        for id, (x, y) in enumerate(val_loader):
            all_label_num += len(y)
            x = x.cuda()
            y = y.cuda()
            if model_type == 'rgbonly':
                x = x[..., :3]
                x = x.permute(0, -1, 1, 2, 3)
            out = model(x)

            loss = criterion(out, y)
            running_loss += loss.item() * y.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == y).sum()
            for idx in range(y.size(0)):
                if pred[idx] != y[idx]:
                    error_ids.append(val_dataset.X_path[batch_size*id+idx]+','+str(y[idx].item())+','+str(pred[idx].item()))
            total_correct += num_correct.item()

            if id % 20 == 0:
                print('Test: iter: {}/{}. {}/{}. Acc: {:.6f}.'.format(id, len(val_loader), total_correct, all_label_num,
                                                                      total_correct/all_label_num))

    print("Test: Loss: {:.6f} {}/{} Acc: {:.6f}".format(running_loss/(batch_size*id), total_correct, all_label_num,
                                                       total_correct/all_label_num))
    if args.res_path is None:
        f = open(model_path+'/rwf_error_ids.txt', 'w')
    else:
        f = open(args.res_path, 'w')
    for e in error_ids:
        f.write(e+'\n')
    f.close()