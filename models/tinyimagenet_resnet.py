# Based on code taken from https://github.com/weiaicunzai/pytorch-cifar100, https://github.com/ganguli-lab/Synaptic-Flow

"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from models import base, layers


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            layers.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            layers.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            layers.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            layers.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = layers.Identity2d(in_channels)

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                layers.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
                              bias=False),
                layers.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        width = int(out_channels * (base_width / 64.))
        self.residual_function = nn.Sequential(
            layers.Conv2d(in_channels, width, kernel_size=1, bias=False),
            layers.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            layers.Conv2d(width, width, stride=stride, kernel_size=3, padding=1, bias=False),
            layers.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            layers.Conv2d(width, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            layers.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = layers.Identity2d(in_channels)

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                layers.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1,
                              bias=False),
                layers.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(base.base_model):

    def __init__(self, block, num_block, base_width, num_classes=200, dense_classifier=False,
                 block_size=[64, 128, 256, 512, 512], resopt=True):
        super().__init__()

        self.in_channels = 64
        self.conv1 = self._res_optimizer(resopt)

        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, block_size[0], num_block[0], 1, base_width)
        self.conv3_x = self._make_layer(block, block_size[1], num_block[1], 2, base_width)
        self.conv4_x = self._make_layer(block, block_size[2], num_block[2], 2, base_width)
        self.conv5_x = self._make_layer(block, block_size[3], num_block[3], 2, base_width)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = layers.Linear(block_size[4] * block.expansion, num_classes)
        if dense_classifier:
            self.fc = nn.Linear(block_size[4] * block.expansion, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layers.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layers.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, base_width):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            layer_list.append(block(self.in_channels, out_channels, stride, base_width))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layer_list)

    def _res_optimizer(self, resopt):
        if resopt is True:
            return nn.Sequential(layers.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                                 layers.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        else:
            return nn.Sequential(layers.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                                 layers.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def _resnet(arch, block, num_block, base_width, num_classes, dense_classifier, pretrained, block_size, resopt):
    model = ResNet(block, num_block, base_width, num_classes, dense_classifier, block_size, resopt)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-cifar{}.pt'.format(arch, num_classes)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet18(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 18 object
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[64, 128, 256, 512, 512], resopt=True)


def resnet18_rwd_st36(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet sl 1 object
    """
    return _resnet('resnet18-st-1', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[58, 113, 216, 398, 398], resopt=True)


def resnet18_rwd_st59(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet sl 2 object
    """
    return _resnet('resnet18-st-2', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[54, 102, 184, 305, 305], resopt=True)


def resnet18_rwd_st79(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet sl 3 object
    """
    return _resnet('resnet18-st-3', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[49, 85, 140, 198, 198], resopt=True)

def resnet18dbl(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 18 object
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[128, 256, 512, 1024, 1024], resopt=True)

def resnet18dbl_rwd_sp36(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 18 object
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[116, 224, 427, 796, 796], resopt=True)


def resnet18dbl_rwd_sp59(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 18 object
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[104, 192, 353, 618, 618], resopt=True)


def resnet18dbl_rwd_sp79(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 18 object
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes, dense_classifier, pretrained,
                   block_size=[86, 148, 262, 432, 432], resopt=True)

def resnet50(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 50 object
    """
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64, num_classes, dense_classifier, pretrained,
                   block_size=[64, 128, 256, 512, 512], resopt=True)


def resnet50_rwd_sp36(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 50 object
    """
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64, num_classes, dense_classifier, pretrained,
                   block_size=[58, 110, 210, 402, 402], resopt=True)


def resnet50_rwd_sp59(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 50 object
    """
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64, num_classes, dense_classifier, pretrained,
                   block_size=[48, 90, 164, 330, 330], resopt=True)


def resnet50_rwd_sp79(input_shape, num_classes, dense_classifier=False, pretrained=False):
    """ return a ResNet 50 object
    """
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64, num_classes, dense_classifier, pretrained,
                   block_size=[36, 60, 110, 246, 246], resopt=True)
