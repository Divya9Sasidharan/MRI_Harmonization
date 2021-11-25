import torch.nn as nn

class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, without_activation=False):
        super(InConv, self).__init__()
        if (without_activation):
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), )

        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                      nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)