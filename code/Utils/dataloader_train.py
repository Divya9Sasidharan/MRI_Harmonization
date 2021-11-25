import torch
print(torch.__version__)
import torch.utils.data as Data
from Utils import dataloader

class IntramodalDatasetwithPreprocess(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, t1_filenames, t2_filenames, iterations=1):
        'Initialization'
        self.t1_filenames = t1_filenames
        self.t2_filenames = t2_filenames
        self.norm = False
        self.iterations = iterations

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.t1_filenames) * self.iterations

    def __getitem__(self, idx):
        img_A = dataloader.imgnorm(dataloader.load_3D(self.t1_filenames[idx]))
        img_B = dataloader.imgnorm(dataloader.load_3D(self.t2_filenames[idx]))
        return img_A, img_B

    def training_generator(self):
        t1_train_path, t2_train_path, file_names_train_t1, file_names_train_t2, file_names_val_t1, file_names_val_t2 = dataloader.DataLoader().data
        training_generator = Data.DataLoader(
            IntramodalDatasetwithPreprocess(file_names_train_t1, file_names_train_t1, 1), batch_size=5, shuffle=True,
            drop_last=True)
        return training_generator
