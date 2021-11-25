import glob, os
import nibabel as nb
import numpy as np
import SimpleITK as sitk


def CreateDatasets(t1_train_path, t2_train_path,t1_val_path,t2_val_path):
    file_names_train_t1 = sorted(glob.glob(os.path.join(t1_train_path, "*.nii.gz")))
    file_names_train_t2 = sorted(glob.glob(os.path.join(t2_train_path, "*.nii.gz")))
    file_names_val_t1 = sorted(glob.glob(os.path.join(t1_val_path, "*.nii.gz")))
    file_names_val_t2 = sorted(glob.glob(os.path.join(t2_val_path, "*.nii.gz")))
    return t1_train_path, t2_train_path, file_names_train_t1, file_names_train_t2,file_names_val_t1,file_names_val_t2


def load_3D(name):
    X_nb = nb.load(name)
    X_np = X_nb.dataobj
    model_np = np.zeros(shape=(1, 256, 256))
    if len(X_np.shape) == 2:
        model_np = np.expand_dims(X_np, axis=0)
    else:
        model_np[:, :, :] = X_np[:, :, :]
    return model_np


def imgnorm(N_I, index1=0.0001, index2=0.0001):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1 * len(I_sort))]
    I_max = I_sort[-int(index2 * len(I_sort))]
    N_I = 1.0 * (N_I - I_min) / (I_max - I_min)
    N_I[N_I > 1.0] = 1.0
    N_I[N_I < 0.0] = 0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2

def Norm_Zscore(img):
    img= (img-np.mean(img))/np.std(img)
    return img

def save_img(I_img,savename):
    I2 = sitk.GetImageFromArray(I_img,isVector=False)
    sitk.WriteImage(I2,savename)

def create_full_brain_mask(img_np):
  img_cp_np = np.copy(img_np)
  image_shape = img_cp_np.shape
  min_intensity = np.min(img_cp_np)
  max_intensity = np.max(img_cp_np)
  img_cp_np[img_cp_np > 0.0] = 1.0
  img_cp_np[img_cp_np <= 0.0] = 0.0
  return img_cp_np

class DataLoader:
    def __init__(self):
        self.my_t1_train_path = "../Data/T1_pp_train"
        self.my_t2_train_path = "../Data/T2_pp_train"
        self.my_t1_val_path = "../Data/T1_pp_validation"
        self.my_t2_val_path = "../Data/T2_pp_validation"
        self.data = CreateDatasets(self.my_t1_train_path, self.my_t2_train_path,self.my_t1_val_path,self.my_t2_val_path)
