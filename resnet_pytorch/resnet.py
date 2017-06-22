from torch.autograd import Variable
import torch


class SandwichLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SandwichLayer, self).__init__()
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


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_size, in_channels, out_size, out_channels):
        super(ResNetBlock, self).__init__()
        self.in_size = in_size
        self.in_channels = in_channels
        self.out_size = out_size
        self.out_channels = out_channels

        stride = 1
        if out_size == in_size // 2:
            stride = 2

        self.sandwich1 = SandwichLayer(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=3, stride=stride, padding=1)
        self.sandwich2 = SandwichLayer(in_channels=out_channels, out_channels=out_channels,
                                       kernel_size=3, stride=1, padding=1)
        self.shrink_size = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pad_zeros = torch.nn.ZeroPad2d((self.in_channels // 2, self.in_channels // 2, 0, 0))

    def forward(self, x):
        res_out = self.sandwich1(x)
        res_out = self.sandwich2(res_out)
        residual = x
        # print("resnet output size is", res_out.size())
        # print("residual size is", residual.size())
        if self.out_channels == self.in_channels * 2:
            residual = torch.transpose(residual, 1, 3)
            residual = self.pad_zeros(residual)
            residual = torch.transpose(residual, 1, 3)
        if self.out_size == self.in_size // 2:
            residual = self.shrink_size(residual)

        out = residual + res_out
        return out


class ResNetStage(torch.nn.Module):
    def __init__(self, in_size, in_channels, out_size, out_channels, repeat_times):
        super(ResNetStage, self).__init__()

        layers = []
        for i in range(repeat_times):
            if i == 0:
                layers.append(ResNetBlock(in_size, in_channels, out_size, out_channels))
            else:
                layers.append(ResNetBlock(out_size, out_channels, out_size, out_channels))

        self.stage_model = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.stage_model(x)
        return out


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = SandwichLayer(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2x = ResNetStage(in_size=32, in_channels=32, out_size=32, out_channels=64, repeat_times=2)
        self.conv3x = ResNetStage(in_size=32, in_channels=64, out_size=32, out_channels=128, repeat_times=2)
        self.conv4x = ResNetStage(in_size=32, in_channels=128, out_size=16, out_channels=256, repeat_times=2)
        self.conv5x = ResNetStage(in_size=16, in_channels=256, out_size=8, out_channels=512, repeat_times=2)
        self.avgpool = torch.nn.AvgPool2d(8)
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2x(x)
        x = self.conv3x(x)
        x = self.conv4x(x)
        x = self.conv5x(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.fc(x)

        return x
