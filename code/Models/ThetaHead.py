import torch.nn as nn


class ThetaHead(nn.Module):
    def __init__(self, in_channel, Ct=1):
        self.in_channel = in_channel
        self.Ct = Ct

        super(ThetaHead, self).__init__()
        self.td = self.theta_head(self.in_channel)

    def theta_head(self, in_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=64, stride=16, padding=1, bias=True),
            nn.Flatten(),
            nn.Linear(2704, self.Ct))

        """
        nn.Conv2d(in_channels=4, out_channels = 8, kernel_size=16, stride=4, padding=5, bias=True),
                    nn.Conv2d(in_channels=8, out_channels = 16, kernel_size=8, stride=4, padding=4, bias=True),
        nn.BatchNorm2d(32),
                    nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=5, stride=4, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
        """
        return layer

    def forward(self, x):
        theta_op = self.td(x)
        return theta_op
