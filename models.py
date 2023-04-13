import torch
import torch.nn as nn
from torch.nn import functional as F

import math

import torch
import torch.nn as nn

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.fc1 = nn.Linear(128 * 3 * 3, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128*3*3)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)
        return x, y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out

class A_mobileNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, num_classes):
        super(A_mobileNet, self).__init__()

        # Standard convolution layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU6(inplace=True)

        # DSC layers
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU6(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU6(inplace=True)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU6(inplace=True)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU6(inplace=True)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU6(inplace=True)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(128)
        self.relu7 = nn.ReLU6(inplace=True)

        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, groups=128)
        self.bn8 = nn.BatchNorm2d(256)
        self.relu8 = nn.ReLU6(inplace=True)
        self.cbam1 = CBAM(256)

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        self.bn9 = nn.BatchNorm2d(256)
        self.relu9 = nn.ReLU6(inplace=True)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn10 = nn.BatchNorm2d(256)
        self.relu10 = nn.ReLU6(inplace=True)

        self.conv11 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, groups=256)
        self.bn11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU6(inplace=True)
        self.cbam2 = CBAM(512)

        # Improved DSC layers
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU6(inplace=True)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.bn13 = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU6(inplace=True)

        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn14 = nn.BatchNorm2d(512)
        self.relu14 = nn.ReLU6(inplace=True)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.bn15 = nn.BatchNorm2d(512)
        self.relu15 = nn.ReLU6(inplace=True)

        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn16 = nn.BatchNorm2d(512)
        self.relu16 = nn.ReLU6(inplace=True)
        self.conv17 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.bn17 = nn.BatchNorm2d(512)
        self.relu17 = nn.ReLU6(inplace=True)

        self.conv18 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.bn18 = nn.BatchNorm2d(512)
        self.relu18 = nn.ReLU6(inplace=True)
        self.conv19 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.bn19 = nn.BatchNorm2d(512)
        self.relu19 = nn.ReLU6(inplace=True)

        self.conv20 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512)
        self.bn20 = nn.BatchNorm2d(512)
        self.relu20 = nn.ReLU6(inplace=True)
        self.cbam3 = CBAM(512)

        self.conv21 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)
        self.bn21 = nn.BatchNorm2d(1024)
        self.relu21 = nn.ReLU6(inplace=True)

        self.conv22 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, groups=1024)
        self.bn22 = nn.BatchNorm2d(1024)
        self.relu22 = nn.ReLU6(inplace=True)
        self.conv23 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.bn23 = nn.BatchNorm2d(1024)
        self.relu23 = nn.ReLU6(inplace=True)
        self.cbam4 = CBAM(1024)

        # AvgPool and FC layers
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(1024, 7)
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.cbam1(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        x = self.cbam2(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu13(x)

        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu14(x)
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.relu15(x)

        x = self.conv16(x)
        x = self.bn16(x)
        x = self.relu16(x)
        x = self.conv17(x)
        x = self.bn17(x)
        x = self.relu17(x)

        x = self.conv18(x)
        x = self.bn18(x)
        x = self.relu18(x)
        x = self.conv19(x)
        x = self.bn19(x)
        x = self.relu19(x)

        x = self.conv20(x)
        x = self.bn20(x)
        x = self.relu20(x)
        x = self.cbam3(x)

        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu21(x)

        x = self.conv22(x)
        x = self.bn22(x)
        x = self.relu22(x)
        x = self.conv23(x)
        x = self.bn23(x)
        x = self.relu23(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = self.fc2(x)

        return x, y

__factory = {
    'cnn': ConvNet,
    'A_mobileNet': A_mobileNet,
}

def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)

if __name__ == '__main__':
    pass