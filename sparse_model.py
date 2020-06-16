from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models

class MASConv(nn.Module):
    # Convolution layer for sparse data
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MASConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              groups=groups, dilation=dilation, bias=False)
        self.if_bias = bias
        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x, m = input
        x = x * m
        x = self.conv(x)
        #
        k = self.conv.kernel_size[0]
        weights = torch.ones(torch.Size([1, 1, k, k])).to(x.get_device())
        mc = F.conv2d(m, weights, bias=None, stride=self.conv.stride,
                      padding=self.conv.padding, dilation=self.conv.dilation)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc
        x = x * mc

        if self.if_bias:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)

        return x, m

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return MASConv(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=bias)


class BottleneckSparse(nn.Module):
    def __init__(self, inplanes, outplanes, use_norm=False):
        super(BottleneckSparse, self).__init__()
        width = int(outplanes * 0.5)
        bias = False if use_norm is None else True
        self.use_norm = use_norm
        self.conv1 = conv1x1(inplanes, width, bias)
        if use_norm:
            self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride=1, groups=1, dilation=1, bias=bias)
        if use_norm:
            self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, outplanes, bias)
        if use_norm:
            self.bn3 = nn.BatchNorm2d(outplanes)

        self.expand = None
        if inplanes != outplanes:
            self.expand = conv1x1(inplanes, outplanes, bias)
        if self.expand is not None and use_norm:
            self.expand_bn = nn.BatchNorm2d(outplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x, m = input
        identity = x

        out = self.conv1(x)
        if self.use_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out, m = self.conv2((out, m))
        if self.use_norm:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.use_norm:
            out = self.bn3(out)

        if self.expand is not None:
            identity = self.expand(identity)
            if self.use_norm:
                identity = self.expand_bn(identity)

        out += identity
        out = self.relu(out)

        return out, m

class Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, use_norm=False):
        super(Bottleneck, self).__init__()
        width = int(outplanes * 0.5)
        bias = False if use_norm is None else True
        self.use_norm = use_norm
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=bias)
        if use_norm:
            self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=bias)
        if use_norm:
            self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, stride=1, bias=bias)
        if use_norm:
            self.bn3 = nn.BatchNorm2d(outplanes)

        self.expand = None
        if inplanes != outplanes:
            self.expand = conv1x1(inplanes, outplanes, bias)
        if self.expand is not None and use_norm:
            self.expand_bn = nn.BatchNorm2d(outplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.use_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_norm:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.use_norm:
            out = self.bn3(out)

        if self.expand is not None:
            identity = self.expand(identity)
            if self.use_norm:
                identity = self.expand_bn(identity)

        out += identity
        out = self.relu(out)

        return out

class MASPool(nn.Module):
    # Convolution layer for sparse data
    def __init__(self, in_channels, kernel_size=2, stride=2, padding=0):
        super(MASPool, self).__init__()
        # self.down = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.down = MASConv(in_channels, in_channels, 3, 2, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.maxpool.require_grad = False

    def forward(self, input):
        x, m = input
        x, m = self.down((x,m))
        m = self.maxpool(m)
        return x, m

class ResMod(nn.Module):
    def __init__(self, cin, cout, reps=5):
        super(ResMod, self).__init__()
        self.expand = BottleneckSparse(cin, cout, use_norm=False)
        self.pool = MASPool(cout, 2, 2)
        self.layers = nn.ModuleList([BottleneckSparse(cout, cout, use_norm=False) for _ in range(reps)])

    def forward(self, input):
        x, m = input
        x, m = self.expand((x, m))
        x, m = self.pool((x, m))

        for L in self.layers:
            x, m = L((x, m))

        return x, m

class Encoder(nn.Module):
    def __init__(self, cin1, cin2, scales=4, base_width=32):
        super(Encoder, self).__init__()
        self.scales = scales
        self.path1 = nn.ModuleList()
        self.path2 = nn.ModuleList()
        for s in range(self.scales):
            in1 = cin1 if s == 0 else base_width*s
            in2 = cin2 if s == 0 else base_width*s
            self.path1.append(ResMod(in1, base_width*(s+1)))
            self.path2.append(ResMod(in2, base_width*(s+1)))

    def forward(self, input):
        x1, m1, x2, m2 = input
        endpoints1, endpoints2 = [], []
        for s in range(self.scales):
            x1, m1 = self.path1[s]((x1, m1))
            x2, m2 = self.path2[s]((x2, m2))
            endpoints1.append((x1, m1))
            endpoints2.append((x2, m2))
        return endpoints1, endpoints2

def convt_bn_relu(in_channels, out_channels, kernel_size=3, \
        stride=2, padding=1, output_padding=1, bn=True, relu=True):
    bias = not bn
    layers = [
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias)]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    return layers

class FPymPool(nn.Module):
    def __init__(self, cin, cout):
        super(FPymPool, self).__init__()
        self.poolx2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.poolx4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.conv = Bottleneck(cin, cout, use_norm=True)

    def forward(self, x):
        x1 = x
        x2 = self.poolx2(x)
        x4 = self.poolx4(x)
        x1 = x1
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, scale_factor=4, mode='bilinear', align_corners=False)
        x = x1 + x2 + x4
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, cout, scales, base_width=32):
        super(Decoder, self).__init__()
        self.scales = scales
        self.first = FPymPool(scales*base_width*2, scales*base_width)
        self.mods = nn.ModuleList()
        self.skips = 0
        for s in range(scales-1, 0, -1): # [s-1,...,1]
            planes = s * base_width
            if s > self.skips:
                conv0 = convt_bn_relu((s+1) * base_width, planes)
                conv1 = Bottleneck(2*planes, planes, use_norm=True)
                layers = nn.ModuleList([conv0, conv1])
                self.mods.append(layers)
            else:
                conv0 = convt_bn_relu((s+1) * base_width, planes)
                layers = nn.ModuleList([conv0])
                self.mods.append(layers)
        self.last = convt_bn_relu(base_width, cout, bn=False, relu=False)

    def forward(self, ends1: [], ends2: []):
        assert len(ends1) == len(ends2)
        ends1.reverse()
        ends2.reverse()
        x = None
        i = 0
        for e1, e2 in zip(ends1, ends2):
            x1, m1 = e1[0], e1[1]
            x2, m2 = e2[0], e2[1]
            if x is None:
                x = torch.cat([x1, x2], 1)
                x = self.first(x)
            elif i < self.scales-self.skips-1: # [0,...,s-1-skip-1]
                x = self.mods[i][0](x) # conv0
                x = torch.cat([x, x2], 1)
                x = self.mods[i][1](x) # conv1
                x = x * m2 + x1
                i = i + 1
            else: # [s-1-skip, s-1)
                x = self.mods[i][0](x) # conv0
                x = x + x1
                i = i + 1
        x = self.last(x)
        ends1.reverse()
        ends2.reverse()
        return x

class Model(nn.Module):
    def __init__(self, scales=4, base_width=32, dec_img=False, colorize=False):
        super(Model, self).__init__()
        depth_channels = 4 if colorize else 1
        self.enc = Encoder(depth_channels, 3, scales=scales, base_width=base_width)
        self.dec_depth = Decoder(1, scales=scales, base_width=base_width)
        self.dec_img = dec_img
        if self.dec_img:
            self.dec_img = Decoder(1, scales=scales, base_width=base_width)

    def forward(self, sdepth, mask, img, require_ends=False):
        ends1, ends2 = self.enc((sdepth, mask, img, 1-mask))
        depth = self.dec_depth(ends1, ends2)
        if self.dec_img:
            rgb = self.dec_img(ends1, ends2)
        else:
            rgb = None

        if not require_ends:
            return depth, rgb
        else:
            return depth, rgb, [ends1, ends2]

if __name__ == "__main__":
    dev = 'cpu'
    if torch.has_cuda and torch.cuda.is_available():
        dev = 'cuda'
    depth = torch.rand(2, 1, 512, 256).to(dev)
    mask = torch.rand(2, 1, 512, 256).to(dev)
    img = torch.rand(2, 3, 512, 256).to(dev)
    #
    conv = MASConv(1, 2, kernel_size=3, padding=1).to(dev)
    x, m = conv((depth, mask))
    print('A', x.shape) # [2, 3, 512, 256]
    #
    resb1 = BottleneckSparse(3, 3, use_norm=False).to(dev)
    resb2 = BottleneckSparse(3, 10, use_norm=True).to(dev)
    ret1, m = resb1((img, m))
    ret2, m = resb2((img, m))
    print('B', ret1.shape) # [2, 3, 512, 256]
    print('C', ret2.shape) # [2, 10, 256, 128]
    #
    resm = ResMod(3, 32).to(dev)
    ret, m = resm((img, mask))
    print('D', ret.shape) # [2, 32, 256, 128]
    #
    pool = MASPool(3, 2, 2).to(dev)
    ret, m = pool((img, mask))
    print('E', ret.shape) # [2, 32, 256, 128]
    #
    enc = Encoder(1, 3, scales=4).to(dev)
    ends1, ends2 = enc((depth, mask, img, 1-mask))
    for e in ends1:
        print('F', e[0].shape)
    #
    dec = Decoder(4, scales=4).to(dev)
    ret = dec(ends1, ends2)
    print('G', ret.shape) # [2, 1, 512, 256]
    #
    mod = Model(scales=4).to(dev)
    d, r = mod(depth, mask, img)
    print('H', d.shape) # [2, 1, 512, 256]
    from thop import profile, clever_format
    def profile_model(net, inputs):
        flops, params = profile(mod, inputs, verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(flops, params)
    profile_model(mod, inputs=(depth, mask, img))