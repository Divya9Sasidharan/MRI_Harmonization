import torch.nn as nn
from Models.layers.DoubleConv import DoubleConv

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, maxpool=False):
        super().__init__()
        if (maxpool):
            self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                              DoubleConv(in_channels, out_channels))
        else:
            self.maxpool_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, stride=2, padding=1, kernel_size=4),

                                              nn.BatchNorm2d(in_channels),
                                              nn.LeakyReLU(0.2),

                                              DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)