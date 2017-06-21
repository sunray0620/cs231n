from torch.autograd import Variable
import torch


class SandwichNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SandwichNet, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class BasicBlock(torch.nn.Module):
    def __init__(self, in_size, in_channels, out_size, out_channels):
        super(BasicBlock, self).__init__()

        stride = 1
        if out_size == in_size // 2:
            stride = 2

        self.sandwich1 = SandwichNet(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=3, stride=stride, padding=1)
        self.sandwich2 = SandwichNet(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.sandwich1(x)
        out = self.sandwich2(out)
        out += residual
        return out


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = SandwichNet(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblock1 = BasicBlock(in_size=32, in_channels=16, out_size=32, out_channels=64)
        self.resblock2 = BasicBlock(in_size=32, in_channels=64, out_size=32, out_channels=128)
        self.resblock3 = BasicBlock(in_size=32, in_channels=128, out_size=16, out_channels=256)
        self.resblock4 = BasicBlock(in_size=16, in_channels=256, out_size=8, out_channels=512)

    def forward(self, x):
        print(x.size())     # x ~ [mini_batch, 3, 32, 32]

        # Block conv1
        x = self.conv1(x)
        print(x.size())     # x ~ [mini_batch, 16, 32, 32]

        # Layer conv1
        for i in range(2):
            self.resblock1()

        return x


filter_dim_list = [
                  [((3, 64), (3, 64)), 3],
                  [((3, 128), (3, 128)), 4],
                  [((3, 256), (3, 256)), 6],
                  [((3, 512), (3, 512)), 3]]

