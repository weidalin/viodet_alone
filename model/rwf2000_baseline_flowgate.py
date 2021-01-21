import torchvision
import torch.nn as nn
from torch.nn import init
import torch
import numpy as np
import math

class RWF_FlowGate(nn.Module):
    def __init__(self):
        super(RWF_FlowGate, self).__init__()

        self.relu = nn.ReLU()
        # RGB
        self.conv3dr11 = nn.Conv3d(3, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bnr11 = nn.BatchNorm3d(16)
        self.conv3dr12 = nn.Conv3d(16, 16, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bnr12 = nn.BatchNorm3d(16)
        self.max_poolingr1 = nn.MaxPool3d((1, 2, 2))

        self.conv3dr21 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bnr21 = nn.BatchNorm3d(16)
        self.conv3dr22 = nn.Conv3d(16, 16, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bnr22 = nn.BatchNorm3d(16)
        self.max_poolingr2 = nn.MaxPool3d((1, 2, 2))

        self.conv3dr31 = nn.Conv3d(16, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bnr31 = nn.BatchNorm3d(32)
        self.conv3dr32 = nn.Conv3d(32, 32, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bnr32 = nn.BatchNorm3d(32)
        self.max_poolingr3 = nn.MaxPool3d((1, 2, 2))

        self.conv3dr41 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bnr41 = nn.BatchNorm3d(32)
        self.conv3dr42 = nn.Conv3d(32, 32, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bnr42 = nn.BatchNorm3d(32)
        self.max_poolingr4 = nn.MaxPool3d((1, 2, 2))

        # Flow
        self.conv3df11 = nn.Conv3d(2, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bnf11 = nn.BatchNorm3d(16)
        self.conv3df12 = nn.Conv3d(16, 16, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bnf12 = nn.BatchNorm3d(16)
        self.max_poolingf1 = nn.MaxPool3d((1, 2, 2))

        self.conv3df21 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bnf21 = nn.BatchNorm3d(16)
        self.conv3df22 = nn.Conv3d(16, 16, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bnf22 = nn.BatchNorm3d(16)
        self.max_poolingf2 = nn.MaxPool3d((1, 2, 2))

        self.conv3df31 = nn.Conv3d(16, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bnf31 = nn.BatchNorm3d(32)
        self.conv3df32 = nn.Conv3d(32, 32, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bnf32 = nn.BatchNorm3d(32)
        self.max_poolingf3 = nn.MaxPool3d((1, 2, 2))

        self.conv3df41 = nn.Conv3d(32, 32, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bnf41 = nn.BatchNorm3d(32)
        self.conv3df42 = nn.Conv3d(32, 32, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bnf42 = nn.BatchNorm3d(32)
        self.max_poolingf4 = nn.MaxPool3d((1, 2, 2))

        # Fusion and Pooling
        self.max_pooling5 = nn.MaxPool3d((8, 1, 1))

        # Merging Block
        self.conv3d61 = nn.Conv3d(32, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn61 = nn.BatchNorm3d(64)
        self.conv3d62 = nn.Conv3d(64, 64, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bn62 = nn.BatchNorm3d(64)
        self.max_pooling6 = nn.MaxPool3d((2, 2, 2))

        self.conv3d71 = nn.Conv3d(64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn71 = nn.BatchNorm3d(64)
        self.conv3d72 = nn.Conv3d(64, 64, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bn72 = nn.BatchNorm3d(64)
        self.max_pooling7 = nn.MaxPool3d((2, 2, 2))

        self.conv3d81 = nn.Conv3d(64, 128, (1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn81 = nn.BatchNorm3d(128)
        self.conv3d82 = nn.Conv3d(128, 128, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bn82 = nn.BatchNorm3d(128)
        self.max_pooling8 = nn.MaxPool3d((2, 3, 3))

        # FC Layers
        self.L1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.2)
        self.L2 = nn.Linear(128, 32)
        self.Classify = nn.Linear(32, 2)
        # self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
                # init.normal_(m.weight, std=0.001)
                # init.constant_(m.bias, 0)
                # init.kaiming_normal_(m.weight, mode='fan_out')
                # init.constant_(m.bias, 0)

    def forward(self, x):
        rgb = x[..., :3]
        rgb = rgb.permute(0, -1, 1, 2, 3)
        flow = x[..., 3:]
        flow = flow.permute(0, -1, 1, 2, 3)

        # rgb
        rgb = self.conv3dr11(rgb)
        rgb = self.bnr11(rgb)
        rgb = self.relu(rgb)
        rgb = self.conv3dr12(rgb)
        rgb = self.bnr12(rgb)
        rgb = self.relu(rgb)
        rgb = self.max_poolingr1(rgb)

        rgb = self.conv3dr21(rgb)
        rgb = self.bnr21(rgb)
        rgb = self.relu(rgb)
        rgb = self.conv3dr22(rgb)
        rgb = self.bnr22(rgb)
        rgb = self.relu(rgb)
        rgb = self.max_poolingr2(rgb)

        rgb = self.conv3dr31(rgb)
        rgb = self.bnr31(rgb)
        rgb = self.relu(rgb)
        rgb = self.conv3dr32(rgb)
        rgb = self.bnr32(rgb)
        rgb = self.relu(rgb)
        rgb = self.max_poolingr3(rgb)

        rgb = self.conv3dr41(rgb)
        rgb = self.bnr41(rgb)
        rgb = self.relu(rgb)
        rgb = self.conv3dr42(rgb)
        rgb = self.bnr42(rgb)
        rgb = self.relu(rgb)
        rgb = self.max_poolingr4(rgb)

        # flow
        flow = self.conv3df11(flow)
        flow = self.bnf11(flow)
        flow = self.relu(flow)
        flow = self.conv3df12(flow)
        flow = self.bnf12(flow)
        flow = self.relu(flow)
        flow = self.max_poolingf1(flow)

        flow = self.conv3df21(flow)
        flow = self.bnf21(flow)
        flow = self.relu(flow)
        flow = self.conv3df22(flow)
        flow = self.bnf22(flow)
        flow = self.relu(flow)
        flow = self.max_poolingf2(flow)

        flow = self.conv3df31(flow)
        flow = self.bnf31(flow)
        flow = self.relu(flow)
        flow = self.conv3df32(flow)
        flow = self.bnf32(flow)
        flow = self.relu(flow)
        flow = self.max_poolingf3(flow)

        flow = self.conv3df41(flow)
        flow = self.bnf41(flow)
        flow = self.relu(flow)
        flow = self.conv3df42(flow)
        flow = self.bnf42(flow)
        flow = self.relu(flow)
        flow = self.max_poolingf4(flow)

        # Fusion and Pooling
        x = torch.mul(rgb, flow)
        x = self.max_pooling5(x)

        # Merging Block
        x = self.conv3d61(x)
        x = self.bn61(x)
        x = self.relu(x)
        x = self.conv3d62(x)
        x = self.bn62(x)
        x = self.relu(x)
        x = self.max_pooling6(x)

        x = self.conv3d71(x)
        x = self.bn71(x)
        x = self.relu(x)
        x = self.conv3d72(x)
        x = self.bn72(x)
        x = self.relu(x)
        x = self.max_pooling7(x)

        x = self.conv3d81(x)
        x = self.bn81(x)
        x = self.relu(x)
        x = self.conv3d82(x)
        x = self.bn82(x)
        x = self.relu(x)
        x = self.max_pooling8(x)

        x = x.view(x.size()[0], -1)

        # FC Layers
        x = self.L1(x)
        # x = self.relu(x)
        x = self.dropout(x)
        x = self.L2(x)
        # x = self.relu(x)
        pred = self.Classify(x)
        # pred = self.softmax(x)

        return pred