import wandb
import os
from Models.CompleteModel import CompleteModel
import torch.nn as nn
import torch
'''print(os.getcwd())
os.chdir('../code')
print(os.getcwd())'''
os.environ["WANDB_API_KEY"] = "6bfc1b7e06e00a76fec81812a8c77640ed185e4e"
wandb.login()

'''#---training---'''
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
