import torch
import torch.nn as nn
import numpy as np

class RandomizationHead(nn.Module):
    def __init__(self, Cb=7, Ct=1, randomize_flag=True, img_dim=256):
        self.Cb = Cb
        self.Ct = Ct
        self.randomize_flag = randomize_flag
        self.img_dim = img_dim
        print(self.Cb)
        super(RandomizationHead, self).__init__()
        self.feature_map_selection_lst_recon1 = self.randomize_betas()
        self.feature_map_selection_lst_recon2 = self.randomize_betas()
        self.feature_map_selection_lst_recon3 = self.randomize_betas()
        self.feature_map_selection_lst_recon4 = self.randomize_betas()
        print("beta_selection1: {}".format(self.feature_map_selection_lst_recon1))
        print("beta_selection2: {}".format(self.feature_map_selection_lst_recon2))
        print("beta_selection3: {}".format(self.feature_map_selection_lst_recon3))
        print("beta_selection4: {}".format(self.feature_map_selection_lst_recon4))

        self.theta_selection_lst_recon = self.randomize_thetas()
        print("theta selction: {}".format(self.randomize_thetas()))

    def randomize_betas(self):

        """
        - There are two betas each from image 1 and image 2 = B1, B2
        - In each B1 and B2 there are 5 channels of (128*128) same size as input image
        - For each Beta channel do a coin toss(if there are 5 channels then 5 coin tosses)
        - For each channel either pick the channel information from B1 or B2 with equal probability
        - We have a total of 5 channels from B1 and 5 channels from B2 concatenated to 10 channels tensor
        - [0,1,2,3,4] = B1; [5,6,7,8,9] = B2 ===> concatenated_Beta=[0,1,2,3,4,   5,6,7,8,9]
        - For channel1 pick between (0,5), channel2 pick between (1,6), channel3 pick between(2,7), channel4 pick between(3,8), channel5 pick between(4, 9) with equal probability
        - So there are 2^5 = 32 possible combinations=> Same is mentioned in youtube video

        """

        rnd_lst = np.random.choice(2, self.Cb, p=[0.5, 0.5]).tolist()
        print(rnd_lst)
        # [0,0,1,0,1] ---> [0, 1, 7, 3, 9]
        for i in range(len(rnd_lst)):
            ln = len(rnd_lst) - 1
            if (rnd_lst[i] == 0):
                rnd_lst[i] = rnd_lst[i] + i
            else:
                rnd_lst[i] = rnd_lst[i] + ln + i

        return rnd_lst

    def randomize_thetas(self):
        """
        Theta would also be chosen randomly among given two theta
        this helps n auto encoding. If 0 then pick X else pick Y
        """

        random_theta = np.random.choice(2, 4, p=[0.5, 0.5]).tolist()
        return random_theta

    def forward(self, beta1, beta2, theta1, theta2):

        if self.randomize_flag:

            # Get 4 randomized maps selection for 4 decoder instances
            # print("selected randomized flag")

            self.feature_map_selection_lst_recon1 = self.randomize_betas()
            self.feature_map_selection_lst_recon2 = self.randomize_betas()
            self.feature_map_selection_lst_recon3 = self.randomize_betas()
            self.feature_map_selection_lst_recon4 = self.randomize_betas()

            self.theta_selection_lst_recon = self.randomize_thetas()
            # print(self.feature_map_selection_lst_recon1, self.feature_map_selection_lst_recon2, self.feature_map_selection_lst_recon3, self.feature_map_selection_lst_recon4)

            # Reconstruct theta to match the dimensionality of beta to 128*128
            batch_size = beta1.shape[0]
            theta1_reconstructed = theta1.repeat(1, self.img_dim, self.img_dim).reshape(batch_size, self.Ct,
                                                                                        self.img_dim, self.img_dim)
            theta2_reconstructed = theta2.repeat(1, self.img_dim, self.img_dim).reshape(batch_size, self.Ct,
                                                                                        self.img_dim, self.img_dim)
            theta_reconstructed = [theta1_reconstructed, theta2_reconstructed]

            # Use the randomized map selection to select  channels from both betas
            # Use randomized thetas to autoencode
            beta = torch.cat((beta1, beta2), dim=1)

            beta_recon1 = beta[:, self.feature_map_selection_lst_recon1, :, :]
            beta_recon2 = beta[:, self.feature_map_selection_lst_recon2, :, :]
            beta_recon3 = beta[:, self.feature_map_selection_lst_recon3, :, :]
            beta_recon4 = beta[:, self.feature_map_selection_lst_recon4, :, :]

            # Combine spatially reconstructed randomized theta and randomized beta feature maps

            input_recon1 = torch.cat((beta_recon1, theta_reconstructed[self.theta_selection_lst_recon[0]]), 1)
            input_recon2 = torch.cat((beta_recon2, theta_reconstructed[self.theta_selection_lst_recon[1]]), 1)
            input_recon3 = torch.cat((beta_recon3, theta_reconstructed[self.theta_selection_lst_recon[2]]), 1)
            input_recon4 = torch.cat((beta_recon4, theta_reconstructed[self.theta_selection_lst_recon[3]]), 1)

        else:
            # Reconstruct theta to match the dimensionality of beta to 128*128
            batch_size = beta1.shape[0]
            theta1_reconstructed = theta1.repeat(1, self.img_dim, self.img_dim).reshape(batch_size, self.Ct,
                                                                                        self.img_dim, self.img_dim)
            theta2_reconstructed = theta2.repeat(1, self.img_dim, self.img_dim).reshape(batch_size, self.Ct,
                                                                                        self.img_dim, self.img_dim)

            input_recon1 = torch.cat((beta1, theta1_reconstructed), 1)
            input_recon2 = torch.cat((beta1, theta2_reconstructed), 1)
            input_recon3 = torch.cat((beta2, theta1_reconstructed), 1)
            input_recon4 = torch.cat((beta2, theta2_reconstructed), 1)

        return input_recon1, input_recon2, input_recon3, input_recon4, self.theta_selection_lst_recon
