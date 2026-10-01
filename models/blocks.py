import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size, stride, padding)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        return self.conv2(x)


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(d_model=input_dim // 4, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        chunks = torch.chunk(x_norm, 4, dim=2)
        x_mamba = torch.cat([self.mamba(c) + self.skip_scale * c for c in chunks], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        return x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.BN(self.conv(x)))


class BFEB(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map, pred):
        pred = self.sigmoid(pred)
        b_att = 1 - torch.abs(pred - 0.5) / 0.5
        f_att = torch.abs(0.5 - pred) - b_att
        fi = f_att - b_att
        return feature_map * fi + feature_map


class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.BN = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, y, alpha=0.5):
        x1, y1 = x, y
        x = self.relu(self.BN(self.conv(x))) + y1
        y = self.relu(self.BN(self.conv(y))) + x1
        return alpha * x + (1 - alpha) * y


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.GELU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )

    def forward(self, x):
        size = (x.size(2), x.size(3))
        att = self.mlp(F.avg_pool2d(x, size, stride=size)) + self.mlp(F.max_pool2d(x, size, stride=size))
        scale = torch.sigmoid(att).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = torch.sigmoid(self.spatial(x_compress))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        return self.SpatialGate(self.ChannelGate(x))


class PASPP(nn.Module):
    def __init__(self, inplanes, outplanes, output_stride=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        dilations = {4: [1, 6, 12, 18], 8: [1, 4, 6, 10], 2: [1, 12, 24, 36], 16: [1, 2, 3, 4], 1: [1, 16, 32, 48]}
        if output_stride not in dilations:
            raise NotImplementedError
        d = dilations[output_stride]
        self._norm_layer = norm_layer
        self.silu = nn.SiLU(inplace=True)
        c = inplanes // 4
        self.conv1 = self._make_layer(inplanes, c)
        self.conv2 = self._make_layer(inplanes, c)
        self.conv3 = self._make_layer(inplanes, c)
        self.conv4 = self._make_layer(inplanes, c)
        self.atrous_conv1 = nn.Conv2d(c, c, kernel_size=3, dilation=d[0], padding=d[0])
        self.atrous_conv2 = nn.Conv2d(c, c, kernel_size=3, dilation=d[1], padding=d[1])
        self.atrous_conv3 = nn.Conv2d(c, c, kernel_size=3, dilation=d[2], padding=d[2])
        self.atrous_conv4 = nn.Conv2d(c, c, kernel_size=3, dilation=d[3], padding=d[3])
        self.conv5 = self._make_layer(inplanes // 2, inplanes // 2)
        self.conv6 = self._make_layer(inplanes // 2, inplanes // 2)
        self.convout = self._make_layer(inplanes, inplanes)

    def _make_layer(self, inplanes, outplanes):
        return nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1), self._norm_layer(outplanes), self.silu)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x12 = x1 + x2
        x34 = x3 + x4

        x1 = self.atrous_conv1(x1) + x12
        x2 = self.atrous_conv2(x2) + x12
        x3 = self.atrous_conv3(x3) + x34
        x4 = self.atrous_conv4(x4) + x34

        x12 = self.conv5(torch.cat([x1, x2], dim=1))
        x34 = self.conv5(torch.cat([x3, x4], dim=1))
        return self.convout(torch.cat([x12, x34], dim=1))
