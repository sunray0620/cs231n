from torch.autograd import Variable
import torch
# import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class SandwichNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(SandwichNet, self).__init__()
        self.conv = conv3x3(in_channels, out_channels, stride)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.sandwich1 = SandwichNet(in_channels, out_channels, stride)
        self.sandwich2 = SandwichNet(in_channels, out_channels, stride)

    def forward(self, x):
        residual = x
        out = self.sandwich1(x)
        out = self.sandwich2(out)
        out += residual
        return out


class ResNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        self.in_channel = 64
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        print(x.size())
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        return x



