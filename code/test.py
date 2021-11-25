import torch
from Utils.dataloader_train import IntramodalDatasetwithPreprocess
from Utils.dataloader_test import IntramodalInferenceDatasetwthpreprocess
print(torch.__version__)
import torch.utils.data as Data
from Utils import dataloader, dataloader_train
import torch.nn as nn
import wandb
import os
import piq
from Models.CompleteModel import CompleteModel
#import train as train_module
import variable as var

class test:
    def eval_loss(epoch=1):
        reconstruction_loss_lst = []
        hard_epoch = 300.0
        reconstruction_loss_lst = []
        cosine_similarity_loss_lst = []
        total_loss_lst = []
        ssim_loss_lst = []
        tones = torch.ones(size=(256, 256))  # .to("opencl")
        teps = torch.Tensor([1e-12])  # .to("opencl")
        sim_loss = 0.0
        inference_generator = IntramodalInferenceDatasetwthpreprocess.inference_generator(self=None)
        for X, Y in inference_generator:
            X = X.float()  # .to("opencl")
            Y = Y.float()  # .to("opencl")
            x_size = X.size()
            batch_size = x_size[0]
            u_x, u_y, d1, d2, d3, d4, o_x, o_y, t_x, t_y, r1, r2, r3, r4, theta_selection_lst_recon =var.complete_model(X,
                                                                                                                     Y,
                                                                                                                     epoch,
                                                                                                                     hard_epoch=hard_epoch)

            tmp_lst = []
            for theta in theta_selection_lst_recon:
                if (theta == 0):
                    tmp_lst.append(X)
                else:
                    tmp_lst.append(Y)

            # Find reconsturction MSE loss between images
            # Here we pass theta 1 and theta 2 alternately hence we should compare with T1 and T2 alternatiely
            recon_loss_1 = var.reconstruction_loss(tmp_lst[0], r1)
            recon_loss_2 = var.reconstruction_loss(tmp_lst[1], r2)
            recon_loss_3 = var.reconstruction_loss(tmp_lst[2], r3)
            recon_loss_4 = var.reconstruction_loss(tmp_lst[3], r4)

            fsim_loss_1 = 1 - piq.fsim(tmp_lst[0], r1, data_range=1.0, chromatic=False)
            fsim_loss_2 = 1 - piq.fsim(tmp_lst[1], r2, data_range=1.0, chromatic=False)
            fsim_loss_3 = 1 - piq.fsim(tmp_lst[2], r3, data_range=1.0, chromatic=False)
            fsim_loss_4 = 1 - piq.fsim(tmp_lst[3], r4, data_range=1.0, chromatic=False)

            recon_loss = ((recon_loss_1 + recon_loss_2 + recon_loss_3 + recon_loss_4) / 4.0)
            fsim_loss = ((fsim_loss_1 + fsim_loss_2 + fsim_loss_3 + fsim_loss_4) / 4.0)
            sim_loss_cosine = 1.0 - var.similarity_loss(o_x.view(batch_size, var.config["Cb"], -1),
                                                    o_y.view(batch_size, var.config["Cb"], -1)).mean()
            sim_loss = sim_loss_cosine
            total_loss = var.hyperparam1 * recon_loss + var.hyperparam2 * var.lmbda * sim_loss
            reconstruction_loss_lst.append(recon_loss.detach().cpu().numpy().item())
            cosine_similarity_loss_lst.append(sim_loss_cosine.detach().cpu().numpy().item())
            total_loss_lst.append(total_loss.detach().cpu().numpy().item())
            ssim_loss_lst.append(fsim_loss.detach().cpu().numpy().item())


            del X, Y, u_x, t_x, u_y, t_y, r1, r2, r3, r4, d1, d2, d3, d4, o_x, o_y, tmp_lst, theta_selection_lst_recon
            # torch.cuda.empty_cache()
            return cosine_similarity_loss_lst, ssim_loss_lst, reconstruction_loss_lst, total_loss_lst
