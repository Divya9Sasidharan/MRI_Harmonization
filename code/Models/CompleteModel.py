import torch.nn as nn

from Models.BetaHead import BetaHead
from Models.RandomizationHead import RandomizationHead
from Models.ReconstructionHead import ReconstructionHead
from Models.ThetaHead import ThetaHead
from Models.UNet import UNet


class CompleteModel(nn.Module):
    def __init__(self, n_channels=1, n_classes=16, bilinear=False, Cb=5, Ct=1, device="cpu", randomize_flag=True,
                 gumbel=True, maxpool=False, tau_min=0.5, without_activation=False):
        self.device = device
        self.Cb = Cb
        self.Ct = Ct
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.randomize_flag = randomize_flag
        self.gumbel = gumbel
        self.maxpool = maxpool
        self.tau_min = tau_min
        self.without_activation = without_activation
        self.use_output_conv = True
        self.use_input_conv = True
        super(CompleteModel, self).__init__()

        self.encoder_unet_model = UNet(self.n_channels, self.n_classes, bilinear=self.bilinear, maxpool=self.maxpool,
                                       without_activation=self.without_activation).to(device)
        self.betahead = BetaHead(self.n_classes, self.Cb).to(device)
        self.thetahead = ThetaHead(self.n_classes, self.Ct).to(device)
        self.randomizehead = RandomizationHead(self.Cb, self.Ct, self.randomize_flag).to(device)
        self.decoder_unet_model = UNet(self.Cb + self.Ct, self.n_classes, bilinear=self.bilinear, maxpool=self.maxpool,
                                       without_activation=self.without_activation).to(device)
        self.reconstructionhead = ReconstructionHead(self.n_classes).to(device)

    def forward(self, X, Y, epoch, hard_flag=True, hard_epoch=200.0):
        X = X.float().to(self.device)
        Y = Y.float().to(self.device)

        x_size = X.size()
        batch_size = x_size[0]

        annelaling_tau = max(1.0 - epoch / hard_epoch, self.tau_min)
        # Send images to encoder unet and get 16 channel output
        # 16 channels from encoder unet fed to betahead to get 5 channels of anatomy
        # Same 16 channels fed to thetahead to produce a vector to represent protocol

        u_x = self.encoder_unet_model(X, use_output_conv=self.use_output_conv)
        p_x, o_x, s_x = self.betahead(u_x, hard_flag=hard_flag, tau=annelaling_tau, gumbel=self.gumbel)
        t_x = self.thetahead(u_x)

        u_y = self.encoder_unet_model(Y, use_output_conv=self.use_output_conv)
        p_y, o_y, s_y = self.betahead(u_y, hard_flag=hard_flag, tau=annelaling_tau, gumbel=self.gumbel)
        t_y = self.thetahead(u_y)

        # randomize Beta and concatenate theta
        input_recon1, input_recon2, input_recon3, input_recon4, theta_selection_lst_recon = self.randomizehead(o_x, o_y,
                                                                                                               t_x, t_y)

        # Pass the concatenated randomized betas with theta to decoder unet to get 16 channel output
        d1 = self.decoder_unet_model(input_recon1, use_output_conv=self.use_output_conv)
        d2 = self.decoder_unet_model(input_recon2, use_output_conv=self.use_output_conv)
        d3 = self.decoder_unet_model(input_recon3, use_output_conv=self.use_output_conv)
        d4 = self.decoder_unet_model(input_recon4, use_output_conv=self.use_output_conv)

        # 16 channel output from decoder unet is fed to reconstruction head to reprocuce input image according to the theta
        # If theta is theta1 then image represents T1 weightesd, else tehta2 represents T2 weighted
        r1 = self.reconstructionhead(d1)
        r2 = self.reconstructionhead(d2)
        r3 = self.reconstructionhead(d3)
        r4 = self.reconstructionhead(d4)
        return u_x, u_y, d1, d2, d3, d4, o_x, o_y, t_x, t_y, r1, r2, r3, r4, theta_selection_lst_recon