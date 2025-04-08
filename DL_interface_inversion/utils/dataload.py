import os
import torch
# import numpy as np
# import torch.nn.functional as fun
from torch.utils.data import Dataset  # For constructing datasets with indexing and length support
# from torch.utils.data import DataLoader

from utils.readGrd import readGrdbynp


class GravityDataset(Dataset):

    def __init__(self, root: str, train: str):  # root specifies root directory, resize customizes data size
        super(GravityDataset, self).__init__()

        self.train = train
        if self.train == 'train':
            self.root = os.path.join(root, 'Train')
        elif self.train == 'test':
            self.root = os.path.join(root, 'Test')
        elif self.train == 'validation':
            self.root = os.path.join(root, 'Validation')

        assert os.path.join(self.root), f"path '{self.root}' does not exists."  # Check if file path exists

        # Get all gravity anomaly files
        dg_names = [i for i in os.listdir(os.path.join(self.root, "dg")) if i.endswith(".grd")]
        self.dg_list = [os.path.join(self.root, "dg", i) for i in dg_names]  # Get all grd files in the dg folder
        self.interface_list = [str(i) for i in range(len(self.dg_list))]
        # self.density = [i for i in range(len(self.dg_list))]
        
        # Match anomalies with corresponding model label data
        for i in range(len(self.dg_list)):
            censtr = self.dg_list[i][-17:-12]  # Extract the middle part of the file number
            # print(censtr)
            # os.system('pause')
            self.interface_list[i] = os.path.join(self.root, 'model', 'SedofBasin_' + censtr + '.grd')

        # Check if files exist
        for i in self.dg_list:  # Note that i in self.dg_list or dg_names is string type not integer
            if os.path.exists(i) is False:
                raise FileExistsError(f"file {i} doesn't exissts.")

    def __getitem__(self, index):  # Get sample and label corresponding to current index in file list
        dg = readGrdbynp(self.dg_list[index])
        label_interface = readGrdbynp(self.interface_list[index])
        
        # Get density number from filename
        numdens = self.dg_list[index][-6:-4]
        label_dens = 0.1 + (int(numdens) - 1) * (0.7 - 0.1) / 40.0
        label_dens = torch.tensor(label_dens)
        # print(numdens, label_dens)
        # os.system('pause')
        
        # Data normalization options (commented out)
        # dg = (dg - np.mean(dg)) / np.std(dg)   # Standardize anomaly to mean=0, std=1
        # dg = (dg - np.min(dg)) / (np.max(dg)-np.min(dg))  # Min-max normalization to [0,1]
        
        # Convert numpy arrays to torch tensors
        label_interface = torch.from_numpy(label_interface).unsqueeze(0)
        dg = torch.from_numpy(dg).unsqueeze(0)
        # print(mod_label.shape,dg.shape)
        dg = dg * (-1.0)  # Invert gravity anomaly sign
        mod_label = label_interface  # Set input/output as float16 type Tensor
        
        # Resampling options (commented out)
        # # Resample data to network target input size 200*200, 200*200, 2D data interpolate function input needs [b,c,h,w]
        # dg_sample = fun.interpolate(dg, size=[64, 64], mode='bilinear', align_corners=True)
        # lab_interface_sample = fun.interpolate(mod_label, size=[64, 64], mode='bilinear', align_corners=True)
        
        # Remove batch dimension
        dg_sample = dg
        label_interface_sample = mod_label
        label_dens = label_dens.unsqueeze(0)
        # print(dg_sample.shape,mod_label_sample.shape)
        # print(scaler)
        return dg_sample.float(), label_interface_sample.float(), label_dens.float()

    def __len__(self):
        return len(self.dg_list)


# # # Test if dataset loading is successful
if __name__ == '__main__':
    myroot = r"D:\Project\DL_interface_inversion\data"
    mydata = GravityDataset(myroot, train='train')
    print(mydata.__len__())
    print(mydata[40][0].shape, mydata[40][1].shape, mydata[40][2].shape)
    print(torch.max(mydata[40][0]))
