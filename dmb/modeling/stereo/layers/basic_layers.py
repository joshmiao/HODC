import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple
from dmb.modeling.stereo.layers.instance_whitening import InstanceWhitening


__all__ = ['conv_bn', 'deconv_bn',
           'conv_bn_relu', 'bn_relu_conv', 'deconv_bn_relu',
           'conv3d_bn', 'deconv3d_bn',
           'conv3d_bn_relu', 'bn_relu_conv3d', 'deconv3d_bn_relu',
           'BasicBlock',
           ]

def consistent_padding_with_dilation(padding, dilation, dim=2):
    assert dim == 2 or dim == 3, 'Convolution layer only support 2D and 3D'
    if dim == 2:
        padding = _pair(padding)
        dilation = _pair(dilation)
    else:  # dim == 3
        padding = _triple(padding)
        dilation = _triple(dilation)

    padding = list(padding)
    for d in range(dim):
        padding[d] = dilation[d] if dilation[d] > 1 else padding[d]
    padding = tuple(padding)

    return padding, dilation


def conv_bn(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            ),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            ),
        )


def deconv_bn(batchNorm, in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, output_padding=output_padding, bias=bias
            ),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, output_padding=output_padding, bias=bias
            ),
        )


def conv3d_bn(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=3)
    if batchNorm:
        return nn.Sequential(
            nn.Conv3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm3d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias
            ),
        )


def deconv3d_bn(batchNorm, in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, output_padding=output_padding, bias=bias),
            nn.BatchNorm3d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, output_padding=output_padding, bias=bias
            ),
        )


def conv_bn_relu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
            nn.ReLU(inplace=True),
        )


def bn_relu_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    if batchNorm:
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
        )
    else:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
        )


def deconv_bn_relu(batchNorm, in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, output_padding=output_padding, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, output_padding=output_padding, bias=bias
            ),
            nn.ReLU(inplace=True),
        )


def conv3d_bn_relu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=3)
    if batchNorm:
        return nn.Sequential(
            nn.Conv3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias
            ),
            nn.ReLU(inplace=True),
        )


def bn_relu_conv3d(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=3)
    if batchNorm:
        return nn.Sequential(
            nn.BatchNorm3d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
        )
    else:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias
            ),
        )


def deconv3d_bn_relu(batchNorm, in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, output_padding=output_padding, bias=bias),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, output_padding=output_padding, bias=bias
            ),
            nn.ReLU(inplace=True),
        )

def conv_in_relu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias),
        nn.InstanceNorm2d(out_planes, affine=False),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, batchNorm, in_planes, out_planes, stride, downsample, padding, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_bn_relu(
            batchNorm=batchNorm, in_planes=in_planes, out_planes=out_planes,
            kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.conv2 = conv_bn(
            batchNorm=batchNorm, in_planes=out_planes, out_planes=out_planes,
            kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return out



class BasicBlock_IN(nn.Module):
    expansion = 1

    def __init__(self, batchNorm, in_planes, out_planes, stride, downsample, padding, dilation):
        super(BasicBlock_IN, self).__init__()
        self.conv1 = conv_bn_relu(
            batchNorm=batchNorm, in_planes=in_planes, out_planes=out_planes,
            kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.conv2 = conv_bn(
            batchNorm=batchNorm, in_planes=out_planes, out_planes=out_planes,
            kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.downsample = downsample
        self.stride = stride
        self.norm = nn.InstanceNorm2d(out_planes, affine=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.norm(out)
        return out


class DomainNorm(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
        self.l2 = l2
        self.weight = nn.Parameter(torch.ones(1,channel,1,1))
        self.bias = nn.Parameter(torch.zeros(1,channel,1,1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True
    def forward(self, x):
        x = self.normalize(x)
        if self.l2:
            x = F.normalize(x, p=2, dim=1)
        return x * self.weight + self.bias
    


def conv_dn_relu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias),
        DomainNorm(out_planes),
        nn.ReLU(inplace=True),
    )

class BasicBlock_DN(nn.Module):
    expansion = 1

    def __init__(self, batchNorm, in_planes, out_planes, stride, downsample, padding, dilation):
        super(BasicBlock_DN, self).__init__()
        self.conv1 = conv_bn_relu(
            batchNorm=batchNorm, in_planes=in_planes, out_planes=out_planes,
            kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.conv2 = conv_bn(
            batchNorm=batchNorm, in_planes=out_planes, out_planes=out_planes,
            kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.downsample = downsample
        self.stride = stride
        self.norm = DomainNorm(out_planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.norm(out)
        return out