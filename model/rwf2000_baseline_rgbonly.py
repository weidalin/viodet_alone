# coding=utf-8
import torchvision
import torch.nn as nn
from torch.nn import init
import torch
import numpy as np
import math

class RWF_RGB(nn.Module):
    def __init__(self):
        super(RWF_RGB, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv3d11 = nn.Conv3d(3, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3d12 = nn.Conv3d(16, 16, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.max_pooling1 = nn.MaxPool3d((1, 2, 2))

        self.conv3d21 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3d22 = nn.Conv3d(16, 16, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.max_pooling2 = nn.MaxPool3d((1, 2, 2))

        self.conv3d31 = nn.Conv3d(16, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3d32 = nn.Conv3d(32, 32, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.max_pooling3 = nn.MaxPool3d((1, 2, 2))

        self.conv3d41 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3d42 = nn.Conv3d(32, 32, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.max_pooling4 = nn.MaxPool3d((1, 2, 2))

        self.max_pooling5 = nn.MaxPool3d((8, 1, 1))

        self.conv3d61 = nn.Conv3d(32, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3d62 = nn.Conv3d(64, 64, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.max_pooling6 = nn.MaxPool3d((2, 2, 2))

        self.conv3d71 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3d72 = nn.Conv3d(64, 64, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.max_pooling7 = nn.MaxPool3d((2, 2, 2))

        self.conv3d81 = nn.Conv3d(64, 128, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv3d82 = nn.Conv3d(128, 128, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.max_pooling8 = nn.MaxPool3d((2, 3, 3))

        self.L1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.2)
        self.L2 = nn.Linear(128, 32)
        self.Classify = nn.Linear(32, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # 这里十分重要，linear层不要初始化！不然跑不出结果！原因不明
            # elif isinstance(m, nn.Linear):
                # init.normal_(m.weight, std=0.001)
                # init.constant_(m.bias, 0)
                # init.kaiming_normal_(m.weight, mode='fan_out')
                # init.constant_(m.bias, 0)
        # init.normal_(self.Classify.weight, std=0.001)
        # init.constant_(self.Classify.bias, 0)

    def forward(self, x):
        x = self.conv3d11(x)
        x = self.relu(x)
        x = self.conv3d12(x)
        x = self.relu(x)
        x = self.max_pooling1(x)

        x = self.conv3d21(x)
        x = self.relu(x)
        x = self.conv3d22(x)
        x = self.relu(x)
        x = self.max_pooling2(x)

        x = self.conv3d31(x)
        x = self.relu(x)
        x = self.conv3d32(x)
        x = self.relu(x)
        x = self.max_pooling3(x)

        x = self.conv3d41(x)
        x = self.relu(x)
        x = self.conv3d42(x)
        x = self.relu(x)
        x = self.max_pooling4(x)

        x = self.max_pooling5(x)

        x = self.conv3d61(x)
        x = self.relu(x)
        x = self.conv3d62(x)
        x = self.relu(x)
        x = self.max_pooling6(x)

        x = self.conv3d71(x)
        x = self.relu(x)
        x = self.conv3d72(x)
        x = self.relu(x)
        x = self.max_pooling7(x)

        x = self.conv3d81(x)
        x = self.relu(x)
        x = self.conv3d82(x)
        x = self.relu(x)
        x = self.max_pooling8(x)

        x = x.view(x.size()[0], -1)

        x = self.L1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.L2(x)
        x = self.relu(x)
        pred = self.Classify(x)

        return pred
