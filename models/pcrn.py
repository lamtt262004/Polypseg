import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from .blocks import PVMLayer, Fusion, CBAM, PASPP, CBR, BFEB, SideoutBlock


class PCRN(nn.Module):
    def __init__(self, c_list=(16, 24, 40, 112, 320), pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b1')
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b1')

        self.ChLayer1 = PVMLayer(c_list[0], c_list[0])
        self.ChLayer2 = PVMLayer(c_list[1], c_list[1])
        self.ChLayer3 = PVMLayer(c_list[2], c_list[2])
        self.ChLayer4 = PVMLayer(c_list[3], c_list[3])
        self.ChLayer5 = PVMLayer(c_list[4], c_list[4])

        self.conv1 = nn.Conv2d(c_list[0], c_list[1], kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(c_list[1], c_list[2], kernel_size=1, padding='same')
        self.conv3 = nn.Conv2d(c_list[2], c_list[3], kernel_size=1, padding='same')
        self.conv4 = nn.Conv2d(c_list[3], c_list[4], kernel_size=1, padding='same')
        self.MP = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fusion1 = Fusion(c_list[1])
        self.fusion2 = Fusion(c_list[2])
        self.fusion3 = Fusion(c_list[3])
        self.fusion4 = Fusion(c_list[4])

        self.cbam1 = CBAM(c_list[0])
        self.cbam2 = CBAM(c_list[1])
        self.cbam3 = CBAM(c_list[2])
        self.cbam4 = CBAM(c_list[3])

        self.b5 = PASPP(c_list[4], c_list[4])

        self.CBR4 = CBR(c_list[4], 1)
        self.CBR3 = CBR(c_list[3], 1)
        self.CBR2 = CBR(c_list[2], 1)
        self.CBR1 = CBR(c_list[1], 1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv11 = nn.Conv2d(c_list[3], 1, kernel_size=3, padding='same')
        self.conv12 = nn.Conv2d(c_list[2], 1, kernel_size=3, padding='same')
        self.conv13 = nn.Conv2d(c_list[1], 1, kernel_size=3, padding='same')
        self.conv14 = nn.Conv2d(c_list[0], 1, kernel_size=3, padding='same')

        self.convup1 = nn.Conv2d(1, c_list[3], kernel_size=3, padding='same')
        self.convup2 = nn.Conv2d(1, c_list[2], kernel_size=3, padding='same')
        self.convup3 = nn.Conv2d(1, c_list[1], kernel_size=3, padding='same')
        self.convup4 = nn.Conv2d(1, c_list[0], kernel_size=3, padding='same')

        self.BFEB = BFEB()
        self.sideout = SideoutBlock(c_list[3], 1, dropout=0.1)

        self.conv_out = nn.Conv2d(c_list[0], 1, kernel_size=1, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        endpoints = self.backbone.extract_endpoints(x)
        p1 = endpoints['reduction_1']
        p2 = endpoints['reduction_2']
        p3 = endpoints['reduction_3']
        p4 = endpoints['reduction_4']
        p5 = endpoints['reduction_5']

        c1 = self.ChLayer1(p1)
        d1 = self.cbam1(c1)
        c1 = self.fusion1(self.conv1(self.MP(c1)), p2)

        c2 = self.ChLayer2(c1)
        d2 = self.cbam2(c2)
        c2 = self.fusion2(self.conv2(self.MP(c2)), p3)

        c3 = self.ChLayer3(c2)
        d3 = self.cbam3(c3)
        c3 = self.fusion3(self.conv3(self.MP(c3)), p4)

        c4 = self.ChLayer4(c3)
        d4 = self.cbam4(c4)
        c4 = self.fusion4(self.conv4(self.MP(c4)), p5)

        c5 = self.ChLayer5(c4)
        d5 = self.b5(c5)

        d5 = self.CBR4(self.up(d5))
        bfeb4 = self.convup1(self.BFEB(self.conv11(d4), d5))
        out2 = self.sigmoid(self.sideout(bfeb4))
        out2 = F.interpolate(out2, size=x.shape[2:], mode='bilinear', align_corners=True)

        d4 = self.CBR3(self.up(bfeb4))
        bfeb3 = self.convup2(self.BFEB(self.conv12(d3), d4))

        d3 = self.CBR2(self.up(bfeb3))
        bfeb2 = self.convup3(self.BFEB(self.conv13(d2), d3))

        d2 = self.CBR1(self.up(bfeb2))
        bfeb1 = self.convup4(self.BFEB(self.conv14(d1), d2))

        out = self.sigmoid(self.conv_out(self.up(bfeb1)))
        return out, out2
