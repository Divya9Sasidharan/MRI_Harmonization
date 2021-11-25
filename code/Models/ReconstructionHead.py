import torch.nn as nn


class ReconstructionHead(nn.Module):
    def __init__(self, in_channel):
        self.in_channel = in_channel
        super(ReconstructionHead, self).__init__()
        self.reconstruction_head = self.reconstruct()

    def reconstruct(self):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=1, stride=1, padding=1, kernel_size=3, bias=False),
            nn.Sigmoid())
        return layer

    def forward(self, input_recon):
        # batch_size = input_recon.shape[0]
        # theta_reconstructed = theta.repeat(1, 128, 128).reshape(batch_size, self.Ct, 128, 128)
        # x_in = torch.cat((beta, theta_reconstructed), 1)
        rh = self.reconstruction_head(input_recon)
        return rh
