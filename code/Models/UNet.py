import torch.nn as nn
from Models.layers.InConv import InConv
from Models.layers.OutConv import OutConv
from Models.layers.DoubleConv import DoubleConv
from Models.layers.Down import Down
from Models.layers.Up import Up


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, maxpool=False, without_activation=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.maxpool = maxpool
        self.without_activation = without_activation

        self.inc1 = InConv(n_channels, 8, self.without_activation)
        self.inc2 = DoubleConv(8, 16)
        self.down1 = Down(16, 32, self.maxpool)
        self.down2 = Down(32, 64, self.maxpool)
        self.down3 = Down(64, 128, self.maxpool)
        self.down4 = Down(128, 256, self.maxpool)
        self.down5 = Down(256, 512, self.maxpool)
        factor = 2 if bilinear else 1
        self.down6 = Down(512, 1024 // factor, self.maxpool)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        self.up5 = Up(64, 32 // factor, bilinear)
        self.up6 = Up(32, 16, bilinear)
        self.outc = OutConv(16, self.n_classes, self.without_activation)

    def forward(self, x, use_input_conv=True, use_output_conv=True):
        x1 = x
        if (use_input_conv):
            x1 = self.inc1(x)
            x1 = self.inc2(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        # print(x5.shape, x6.shape, x7.shape)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)

        if use_output_conv:
            x = self.outc(x)

        return x
