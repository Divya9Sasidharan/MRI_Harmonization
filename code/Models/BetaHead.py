import torch.nn as nn
import torch.nn.functional as F


class BetaHead(nn.Module):
    def __init__(self, in_channel, Cb=5):
        self.in_channel = in_channel
        self.Cb = Cb
        super(BetaHead, self).__init__()
        self.probmaps = self.probability_map(self.in_channel, self.Cb * 1)
        self.softmax_output = nn.Softmax(dim=1)

    def probability_map(self, in_channel, Cb, ks=3, st=1, pd=1, bi=False):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=self.Cb, kernel_size=ks, stride=st, padding=pd, bias=bi))
        return layer

    # nn.BatchNorm2d(self.Cb)
    def forward(self, x, hard_flag=True, tau=1.0, gumbel=True):
        pmap = self.probmaps(x)

        if (gumbel):

            ohe = nn.functional.gumbel_softmax(pmap, tau=tau, hard=hard_flag, dim=1)

        else:
            # channel_positions_for_max_pixels = torch.argmax(pmap, dim=1)
            _, channel_positions_for_max_pixels = pmap.max(dim=1)
            ohe = F.one_hot(channel_positions_for_max_pixels, num_classes=self.Cb)
            ohe = ohe.permute(0, 3, 1,
                              2)  # the channel dim would appear at end 1,256,256,channel ==> permute channel dim to second position 1,channel,256,256
            ohe = ohe.float()

            # print(ohe.shape)
        smax = self.softmax_output(pmap)
        return pmap, ohe, smax
