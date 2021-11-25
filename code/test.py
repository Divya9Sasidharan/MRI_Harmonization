import torch
from Utils.dataloader_test import IntramodalInferenceDatasetwthpreprocess
import warnings
warnings.filterwarnings("ignore")
print(torch.__version__)

from Models.CompleteModel import CompleteModel
import imageio
from PIL import Image
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import os
complete_model_inference = CompleteModel(gumbel=True, maxpool=False, bilinear=True, tau_min=0.5, Cb=4,
                                         without_activation=True)  # .to("cuda")
PATH = '../code/Results/disentangled_latent_space/latent_space_harmonization_tau5_without_maxpool_without_transpose_with_activation_300.pth'
checkpoint = torch.load(PATH,map_location=torch.device('cpu'))
complete_model_inference.load_state_dict(checkpoint)


class test:
    def inference_run1(epoch=1):
        example_number = 0
        beta_np_lst = []
        unet_lst = []
        probmaps_np_lst = []
        inference_generator = IntramodalInferenceDatasetwthpreprocess.inference_generator(self=None)
        for X, Y in inference_generator:
            X = X.float()  # .to("opencl")
            Y = Y.float()  # .to("opencl")
            u_x, u_y, d1, d2, d3, d4, o_x, o_y, t_x, t_y, r1, r2, r3, r4, theta_selection_lst_recon = complete_model_inference(
                X,
                Y,
                epoch)
            print(t_x)
            unet_lst.append(u_x.detach().cpu().numpy())
            print(t_y)
            print(" -============== ============ ============== ")
            print()
            unet_lst.append(u_y.detach().cpu().numpy())
            unet_lst.append(d1.detach().cpu().numpy())
            unet_lst.append(d2.detach().cpu().numpy())
            beta_np_lst.append([o_x.detach().cpu().numpy(), o_y.detach().cpu().numpy()])
            r1_np = r1.detach().cpu().numpy()
            r2_np = r2.detach().cpu().numpy()
            r3_np = r3.detach().cpu().numpy()
            r4_np = r4.detach().cpu().numpy()
            nb.save(nb.Nifti1Image(r1_np, affine=np.eye(4)),
                    '../Results/warped_images/r1_np' + str(example_number) + ".nii.gz")
            nb.save(nb.Nifti1Image(r1_np, affine=np.eye(4)),
                    '../Results/warped_images/r2_np' + str(example_number) + ".nii.gz")
            nb.save(nb.Nifti1Image(r1_np, affine=np.eye(4)),
                    '../Results/warped_images/r3_np' + str(example_number) + ".nii.gz")
            nb.save(nb.Nifti1Image(r1_np, affine=np.eye(4)),
                    '../Results/warped_images/r4_np' + str(example_number) + ".nii.gz")
            imageio.imwrite('../Results/warped_images/r1_np' + str(example_number) + ".jpg",
                            r1_np[0, 0, :, :])
            imageio.imwrite('../Results/warped_images/r2_np' + str(example_number) + ".jpg",
                            r2_np[0, 0, :, :])
            imageio.imwrite('../Results/warped_images/r3_np' + str(example_number) + ".jpg",
                            r3_np[0, 0, :, :])
            imageio.imwrite('../Results/warped_images/r4_np' + str(example_number) + ".jpg",
                            r4_np[0, 0, :, :])
            del X, Y, u_x, t_x, u_y, t_y, r1, r2, r3, r4, d1, d2, d3, d4
            # torch.cuda.empty_cache()
            example_number = example_number + 1

            if (example_number > 6):
                break;
        return beta_np_lst, unet_lst, probmaps_np_lst

os.chdir('../code/Utils')
beta_lst, unet_lst, probmaps_np_lst = test.inference_run1(epoch=1)

image = Image.open('../Results/warped_images/r2_np1.jpg')
data = np.asfarray(image)
plt.figure(figsize=(7, 7))
plt.imshow(data, cmap="Greys_r")
plt.show()
image = Image.open('../Results/warped_images/r1_np1.jpg')
data = np.asfarray(image)
plt.figure(figsize=(7, 7))
plt.imshow(data, cmap="Greys_r")
plt.show()
