import torch
import torch.nn as nn

def Conv2d_3x3(in_size, out_size, stride=1, padding=(1,1)):
    """ Convolution with 3x3 kernel """
    return nn.Conv2d(
            in_size, out_size, kernel_size=3, 
            bias=False, stride=stride, padding=padding)

def Conv2d_1x1(in_size, out_size, stride=1):
    """ Convolution with 1x1 kernel, used when downsampling """
    return nn.Conv2d(
            in_size, out_size, kernel_size=1, 
            bias=False, stride=1)

class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d_3x3(in_size, out_size, stride)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_3x3(out_size, out_size, stride)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.downsample = downsample

    def foward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # downsample when dimensions change/increase
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class Bottleneck(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d_1x1(in_size, in_size, stride)
        self.bn1 = nn.BatchNorm2d(in_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_3x3(in_size, in_size, stride)
        self.bn2 = nn.BatchNorm2d(in_size)
        self.conv3 = Conv2d_1x1(in_size, out_size, stride)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.downsample = downsample

    def foward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, layers, num_classes=1000, block=BasicBlock):
        super(ResNet, self).__init__()
        self.input_size = 64 # starts with 64
        self.layers = layers

        self.conv1 = nn.Conv2d(
                3, self.input_size, kernel_size=7, 
                stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._create_layer(64, layers[0], block=block)
        self.layer2 = self._create_layer(128, layers[1], stride=2, block=block)
        self.layer3 = self._create_layer(256, layers[2], stride=2, block=block)
        self.layer4 = self._create_layer(512, layers[3], stride=2, block=block)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if block is Bottleneck:
            self.fc = nn.Linear(512 * 4, num_classes)
        else:
            self.fc = nn.Linear(512, num_classes)

    def _create_layer(self, output_size, blocks, stride=1, padding=(1,1),
            block=None):
        downsample = None
        # when dimensions change/increase
        if (self.input_size != output_size) or (stride != 1):
            downsample = nn.Sequential(
                    Conv2d_1x1(self.input_size, output_size, stride), 
                    nn.BatchNorm2d(output_size)
            )
        layers = []
        layers.append(block(self.input_size, output_size, downsample=downsample))
        if block is Bottleneck:
            output_size *= 4
        # create the first layer and change the size
        self.input_size = output_size
        for i in range(1, blocks):
            layers.append(block(self.input_size, output_size))

        return nn.Sequential(*layers) 

def _resnet(arch, layers, block):
    model = ResNet(layers, block=block)
    return model

def ResNet18():
    return _resnet('resnet18', [2, 2, 2, 2], BasicBlock)

def ResNet34():
    return _resnet('resnet34', [3, 4, 6, 3], BasicBlock)

def ResNet50():
    return _resnet('resnet50', [3, 4, 6, 3], Bottleneck)

def ResNet101():
    return _resnet('resnet101', [3, 4, 23, 3], Bottleneck)

def ResNet152():
    return _resnet('resnet152', [3, 8, 36, 3], Bottleneck)


if __name__ == '__main__':
    model = ResNet50()
    print(model)
