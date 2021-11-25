import torch
from Utils.dataloader_train import IntramodalDatasetwithPreprocess
from Utils.dataloader_test import IntramodalInferenceDatasetwthpreprocess
from Models.CompleteModel import CompleteModel

print(torch.__version__)
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import wandb
import os
import piq

os.environ["WANDB_API_KEY"] = "6bfc1b7e06e00a76fec81812a8c77640ed185e4e"
wandb.login()
try:
    complete_model
except NameError:
    print("Variables don't exists, hence cannot delete!")
else:
    del complete_model
    # torch.cuda.empty_cache()
    print("Deleted model varialbles")
complete_model = CompleteModel(gumbel=True, maxpool=False, bilinear=True, tau_min=0.5, Cb=4,
                               without_activation=True)  # .to("opencl")
wandb.config = dict(
    epochs=301,
    batch_size=5,
    learning_rate=0.001,
    hyperparam1=1.0,
    hyperparam2=1.0,
    hyperparam3=1.0,
    lmda=0.0005,
    dataset="MRI-IXI-T1,T2",
    maxpool=True, bilinear=True, Cb=4, tau_min=0.5, without_activation=False,
    architecture="UNET-CNN")
config = wandb.config
reconstruction_loss = nn.MSELoss()
l1_similarity_loss = nn.L1Loss()
similarity_loss = torch.nn.CosineSimilarity(dim=2, eps=1e-08)
hyperparam1 = config["hyperparam1"]
hyperparam2 = config["hyperparam2"]
hyperparam3 = config["hyperparam3"]
lmbda = config["lmda"]
lr = config["learning_rate"]
optimizer = torch.optim.Adam(complete_model.parameters(), lr=lr)
model_dir = '../code/Results/disentangled_latent_space'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


class train:
    def fullmodel_one_epoch_run1(epoch=1):
        example_number = 0
        hard_epoch = 300.0
        reconstruction_loss_lst = []
        cosine_similarity_loss_lst = []
        total_loss_lst = []
        ssim_loss_lst = []
        tones = torch.ones(size=(256, 256))  # .to("opencl")
        teps = torch.Tensor([1e-12])  # .to("opencl")
        sim_loss = 0.0
        training_generator = IntramodalDatasetwithPreprocess.training_generator(self=None)
        for X, Y in training_generator:
            X = X.float()  # .to("opencl")
            Y = Y.float()  # .to("opencl")
            x_size = X.size()
            batch_size = x_size[0]
            u_x, u_y, d1, d2, d3, d4, o_x, o_y, t_x, t_y, r1, r2, r3, r4, theta_selection_lst_recon = complete_model(
                X,
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
            recon_loss_1 = reconstruction_loss(tmp_lst[0], r1)
            recon_loss_2 = reconstruction_loss(tmp_lst[1], r2)
            recon_loss_3 = reconstruction_loss(tmp_lst[2], r3)
            recon_loss_4 = reconstruction_loss(tmp_lst[3], r4)
            fsim_loss_1 = 1 - piq.fsim(tmp_lst[0], r1, data_range=1.0, chromatic=False)
            fsim_loss_2 = 1 - piq.fsim(tmp_lst[1], r2, data_range=1.0, chromatic=False)
            fsim_loss_3 = 1 - piq.fsim(tmp_lst[2], r3, data_range=1.0, chromatic=False)
            fsim_loss_4 = 1 - piq.fsim(tmp_lst[3], r4, data_range=1.0, chromatic=False)
            recon_loss = ((recon_loss_1 + recon_loss_2 + recon_loss_3 + recon_loss_4) / 4.0)
            fsim_loss = ((fsim_loss_1 + fsim_loss_2 + fsim_loss_3 + fsim_loss_4) / 4.0)
            sim_loss_cosine = 1.0 - similarity_loss(o_x.view(batch_size, config["Cb"], -1),
                                                    o_y.view(batch_size, config["Cb"], -1)).mean()
            sim_loss = sim_loss_cosine
            total_loss = hyperparam1 * recon_loss + hyperparam2 * lmbda * sim_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            reconstruction_loss_lst.append(recon_loss.detach().cpu().numpy().item())
            cosine_similarity_loss_lst.append(sim_loss_cosine.detach().cpu().numpy().item())
            total_loss_lst.append(total_loss.detach().cpu().numpy().item())
            ssim_loss_lst.append(fsim_loss.detach().cpu().numpy().item())
            del X, Y, u_x, t_x, u_y, t_y, r1, r2, r3, r4, d1, d2, d3, d4, o_x, o_y, tmp_lst, theta_selection_lst_recon
            # torch.cuda.empty_cache()
            example_number = example_number + 1
        cosine_similarity_loss_lst_eval, ssim_loss_lst_eval, reconstruction_loss_lst_eval, total_loss_lst_eval = train.eval_loss(
            epoch)
        if (epoch % 50 == 0):
            if (epoch > 101):
                modelname = model_dir + '/' + "latent_space_harmonization_tau5_without_maxpool_without_transpose_with_activation_" + str(
                    epoch + 0) + '.pth'
                torch.save(complete_model.state_dict(), modelname)
                print("Saving model checkpoints")
            print("epoch: {}".format(epoch + 0))
            print("Losses: {}, {} and {}".format(recon_loss * hyperparam1, sim_loss * hyperparam2 * lmbda,
                                                 total_loss))
            print("Average TRAIN Losses: {}, {}, {}, {}".format(
                sum(abs(x) for x in reconstruction_loss_lst) / len(reconstruction_loss_lst),
                sum(cosine_similarity_loss_lst) / len(cosine_similarity_loss_lst),
                sum(ssim_loss_lst) / len(ssim_loss_lst),
                sum(abs(x) for x in total_loss_lst) / len(total_loss_lst)))
            print("Tau: {}".format(max(config["tau_min"], 1 - epoch / hard_epoch)))
            print()
            print("Average EVAL Losses: {}, {}, {}, {}".format(
                sum(abs(x) for x in reconstruction_loss_lst_eval) / len(reconstruction_loss_lst_eval),
                sum(cosine_similarity_loss_lst_eval) / len(cosine_similarity_loss_lst_eval),
                sum(ssim_loss_lst_eval) / len(ssim_loss_lst_eval),
                sum(abs(x) for x in total_loss_lst_eval) / len(total_loss_lst_eval)))
            print("======= =============== ===========")
            print()

        elif (epoch % 2 == 0):
            print("epoch: {}".format(epoch + 0))
            print("Losses: {}, {} and {}".format(recon_loss * hyperparam1, sim_loss * hyperparam2 * lmbda,
                                                 total_loss))
            print("Average TRAIN Losses: {}, {}, {}, {}".format(
                sum(abs(x) for x in reconstruction_loss_lst) / len(reconstruction_loss_lst),
                sum(cosine_similarity_loss_lst) / len(cosine_similarity_loss_lst),
                sum(ssim_loss_lst) / len(ssim_loss_lst),
                sum(abs(x) for x in total_loss_lst) / len(total_loss_lst)))
            print("Tau: {}".format(max(config["tau_min"], 1 - epoch / hard_epoch)))
            print()
            print("Average EVAL Losses: {}, {}, {}, {}".format(
                sum(abs(x) for x in reconstruction_loss_lst_eval) / len(reconstruction_loss_lst_eval),
                sum(cosine_similarity_loss_lst_eval) / len(cosine_similarity_loss_lst_eval),
                sum(ssim_loss_lst_eval) / len(ssim_loss_lst_eval),
                sum(abs(x) for x in total_loss_lst_eval) / len(total_loss_lst_eval)))
            print("======= =============== ===========")
            print('')
        wandb.log({"epoch": epoch, "total_loss": sum(abs(x) for x in total_loss_lst) / len(total_loss_lst),
                   "cosine_similarity_loss": sum(abs(x) for x in cosine_similarity_loss_lst) / len(
                       cosine_similarity_loss_lst),
                   "fsim_loss": sum(abs(x) for x in ssim_loss_lst) / len(ssim_loss_lst) * hyperparam3,
                   "reconstruction_loss": sum(abs(x) for x in reconstruction_loss_lst) / len(reconstruction_loss_lst),
                   "epoch_eval": epoch,
                   "total_loss_eval": sum(abs(x) for x in total_loss_lst_eval) / len(total_loss_lst_eval),
                   "cosine_similarity_loss_eval": sum(abs(x) for x in cosine_similarity_loss_lst_eval) / len(
                       cosine_similarity_loss_lst_eval),
                   "fsim_loss_eval": sum(abs(x) for x in ssim_loss_lst_eval) / len(
                       ssim_loss_lst_eval) * hyperparam3,
                   "reconstruction_loss_eval": sum(abs(x) for x in reconstruction_loss_lst_eval) / len(
                       reconstruction_loss_lst_eval)})
        return reconstruction_loss_lst, cosine_similarity_loss_lst, ssim_loss_lst, total_loss_lst

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
            u_x, u_y, d1, d2, d3, d4, o_x, o_y, t_x, t_y, r1, r2, r3, r4, theta_selection_lst_recon = complete_model(
                X,
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
            recon_loss_1 = reconstruction_loss(tmp_lst[0], r1)
            recon_loss_2 = reconstruction_loss(tmp_lst[1], r2)
            recon_loss_3 = reconstruction_loss(tmp_lst[2], r3)
            recon_loss_4 = reconstruction_loss(tmp_lst[3], r4)

            fsim_loss_1 = 1 - piq.fsim(tmp_lst[0], r1, data_range=1.0, chromatic=False)
            fsim_loss_2 = 1 - piq.fsim(tmp_lst[1], r2, data_range=1.0, chromatic=False)
            fsim_loss_3 = 1 - piq.fsim(tmp_lst[2], r3, data_range=1.0, chromatic=False)
            fsim_loss_4 = 1 - piq.fsim(tmp_lst[3], r4, data_range=1.0, chromatic=False)

            recon_loss = ((recon_loss_1 + recon_loss_2 + recon_loss_3 + recon_loss_4) / 4.0)
            fsim_loss = ((fsim_loss_1 + fsim_loss_2 + fsim_loss_3 + fsim_loss_4) / 4.0)
            sim_loss_cosine = 1.0 - similarity_loss(o_x.view(batch_size, config["Cb"], -1),
                                                    o_y.view(batch_size, config["Cb"], -1)).mean()
            sim_loss = sim_loss_cosine
            total_loss = hyperparam1 * recon_loss + hyperparam2 * lmbda * sim_loss
            reconstruction_loss_lst.append(recon_loss.detach().cpu().numpy().item())
            cosine_similarity_loss_lst.append(sim_loss_cosine.detach().cpu().numpy().item())
            total_loss_lst.append(total_loss.detach().cpu().numpy().item())
            ssim_loss_lst.append(fsim_loss.detach().cpu().numpy().item())

            del X, Y, u_x, t_x, u_y, t_y, r1, r2, r3, r4, d1, d2, d3, d4, o_x, o_y, tmp_lst, theta_selection_lst_recon
            # torch.cuda.empty_cache()
            return cosine_similarity_loss_lst, ssim_loss_lst, reconstruction_loss_lst, total_loss_lst


with wandb.init(project="MRI-Harmonization", config=config):
    epochs = config["epochs"]
    a = []
    b = []
    c = []
    d = []
    wandb.watch(complete_model, criterion=nn.MSELoss(), log="all", log_freq=10)
    owd = os.getcwd()
    os.chdir('../code/Utils')
    for e in range(epochs):
        m, n, o, p = train.fullmodel_one_epoch_run1(e + 0)
        a.append(m)
        b.append(n)
        c.append(o)
        d.append(p)
