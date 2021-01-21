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
    parser = argparse.ArgumentParser('RWF baseline violence detect model.')
    parser.add_argument('-gpu', type=str, default='0, 1', help='gpu ids to use')
    parser.add_argument('-dataset', type=str, default='rwf')
    parser.add_argument('-data_root', type=str, default='/home/lwg/workspace/Dataset/RWF-2000')
    parser.add_argument('-data_name', type=str, default='RWF-2000-npy-noflow-25') #RWF-2000-npy-withflow --> RWF-2000-npy-noflow-25 edit by weida 2021-1-15
    parser.add_argument('-model_path', type=str, default='weight/rwf_baseline_')
    parser.add_argument('-model_type', type=str, default='rgbonly', choices=['rgbonly', 'flowgate'])
    parser.add_argument('-other_mark', type=str, default=None)

    parser.add_argument('-pre_cut', default=False, action="store_true", help='pre cut 150 frames into smaller snippets')
    parser.add_argument('-pre_cut_len', type=int, default=64, help='pre cut snippets length')
    parser.add_argument('-sample', type=str, default='uniform_sampling',
                        choices=['uniform_sampling', 'random_continuous_sampling', 'random_gap_sampling',
                                 'no_sampling'],
                        help='sampling method for sampling snippet frames to model input')
    parser.add_argument('-target_frames', type=int, default=64, help='model input frames')
    parser.add_argument('-gap', type=int, default=2, help='only for gap sampling')
    parser.add_argument('-flow_type', type=str, default='noflow', choices=['noflow', 'withflow'],
                        help='with flow or no')

    parser.add_argument('-epoch', type=int, default=30, help='number of epochs')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial lr') #0.01 --> 1e-4 update by weida 2021-1-15 
    parser.add_argument('-step_size', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch_size of test_dataloader')
    parser.add_argument('-j', '--num_workers', type=int, default=16, help='worker number.')
    parser.add_argument('-momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-weight_decay', type=float, default=1e-6)
    parser.add_argument('-gamma', type=float, default=0.7)

    parser.add_argument('-logger_path', type=str, default=None, help='logger file')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    save_path = args.model_path
    model_type = args.model_type
    flow_type = args.flow_type
    num_epochs = args.epoch
    num_workers = args.num_workers
    batch_size = args.batch_size
    base_lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    gamma = args.gamma
    step_size = args.step_size
    data_root = args.data_root
    data_type = args.data_name
    train_dir = os.path.join(data_root, data_type, 'train')
    val_dir = os.path.join(data_root, data_type, 'val')
    pre_cut = args.pre_cut
    pre_cut_len = args.pre_cut_len
    sample_method = args.sample
    seq_len = args.target_frames
    gap = args.gap

    save_path += model_type+'_'+flow_type+'_e'+str(num_epochs)+'_s'+str(step_size)+'_b'+str(batch_size)+'_lr'+\
                str(base_lr)+'_g'+str(gamma)+'_'+sample_method+str(seq_len)
    if args.other_mark is not None:
        save_path += args.other_mark
    print('model save path', save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.logger_path is None:
        sys.stdout = Logger(save_path + '/log_train.txt')
    else:
        sys.stdout = Logger(args.logger_path)

    train_dataset = rwf2000_dataset.RWFDataset(directory=train_dir, data_augmentation=True, target_frames=seq_len,
                                               sample=sample_method, gap=gap, pre_cut=pre_cut, pre_cut_len=pre_cut_len)
    val_dataset = rwf2000_dataset.RWFDataset(directory=val_dir, data_augmentation=False, target_frames=seq_len,
                                             sample=sample_method, gap=gap, pre_cut=pre_cut, pre_cut_len=pre_cut_len)
    val_dataset_noprecut = rwf2000_dataset.RWFDataset(directory=val_dir, data_augmentation=False, target_frames=seq_len,
                                                      sample=sample_method, gap=gap)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader_noprecut = torch.utils.data.DataLoader(dataset=val_dataset_noprecut, batch_size=batch_size, shuffle=False,
                                                      num_workers=num_workers)

    if model_type == 'rgbonly':
        model = rwf2000_baseline_rgbonly.RWF_RGB()
    elif model_type == 'flowgate':
        model = rwf2000_baseline_flowgate.RWF_FlowGate()
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()
    print(model)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()

    param = []
    params_dict = dict(model.named_parameters())
    for key, v in params_dict.items():
        param += [{ 'params':v,  'lr_mult':1}]
    optimizer = torch.optim.SGD(param, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    def adjust_lr(epoch):
        lr = base_lr * (gamma ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        return lr

    best_acc = 0
    best_acc_noprecut = 0
    for epoch in range(num_epochs):
        lr = adjust_lr(epoch)
        print('-' * 10)
        print('epoch {}'.format(epoch + 1))

        running_loss = 0.0
        running_acc = 0.0
        start = time.time()
        since = time.time()

        model.train()
        for i, (x, y) in enumerate(train_loader, 1):
            x = x.cuda()
            y = y.cuda()
            if model_type == 'rgbonly':
                x = x[..., :3]
                x = x.permute(0,-1,1,2,3)
            out = model(x)

            loss = criterion(out, y)
            # print 'running_loss: ', running_loss
            running_loss += loss.item() * y.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == y).sum()
            running_acc += num_correct.item()
            # print(out, pred, y, num_correct)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print('Train: [{}/{}] iter: {}/{}. lr: {} . Loss: {:.6f}, Acc: {:.6f} time:{:.1f} s'.format(epoch + 1, num_epochs,
                    i, len(train_loader), lr, running_loss/(batch_size*i), running_acc/(batch_size*i),time.time() - since))
                since = time.time()
        print('Train: Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, running_loss / (len(train_dataset)),
                                                                  running_acc / (len(train_dataset))))
        print('Time:{:.1f} s'.format(time.time() - start))


        def model_eval(epoch, model, val_loader):
            total_correct = 0.0
            all_label_num = 0.0
            model.eval()
            with torch.no_grad():
                for id, (x, y) in enumerate(val_loader):
                    all_label_num += len(y)
                    # print(x.shape)
                    x = x.cuda()
                    y = y.cuda()
                    if model_type == 'rgbonly':
                        x = x[..., :3]
                        x = x.permute(0, -1, 1, 2, 3)
                    out = model(x)
                    _, pred = torch.max(out, 1)
                    num_correct = (pred == y).sum()
                    total_correct += num_correct.item()

                    if id % 20 == 0:
                        print(
                            'Test: [{}/{}] iter: {}/{}. {}/{}. Acc: {:.6f}.'.format(epoch + 1, num_epochs, id,
                                                                                    len(val_loader),
                                                                                    total_correct, all_label_num,
                                                                                    total_correct / all_label_num))
            acc = round(total_correct / all_label_num, 5)
            print("Test: Finish {} epoch, Acc: {:.6f}".format(epoch + 1, acc))
            return acc

        acc = model_eval(epoch, model, val_loader)
        acc_noprecut = model_eval(epoch, model, val_loader_noprecut)
        if acc >= best_acc:
            torch.save(model.state_dict(), save_path + '/epoch{}_{}.pth'.format(epoch+1, acc))
            best_acc = acc
        if acc_noprecut >= best_acc_noprecut:
            torch.save(model.state_dict(), save_path + '/epoch{}_{}_noprecut.pth'.format(epoch+1, acc_noprecut))
            best_acc_noprecut = acc_noprecut