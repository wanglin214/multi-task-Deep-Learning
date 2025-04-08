import os
import torch
from torch.utils.data import DataLoader
from utils import dataload
import tqdm


myroot = r"D:\Project\DL_interface_inversion\data"  # Root directory for data
batchsize = 1  # Batch size for data loading

# Create DataLoader for test data
train_loader = DataLoader(dataload.GravityDataset(myroot, train='test'),
                          batch_size=batchsize,
                          shuffle=False,
                          prefetch_factor=4,
                          num_workers=8,
                          )  # Load test data

dataset_size = len(train_loader)
# print(dataset_size)

if __name__ == '__main__':
    sample_max = 0
    for i, (dg, depth, density) in enumerate(tqdm.tqdm(train_loader)):
        # Check for models with depth exceeding threshold
        # if torch.max(depth)>17.0:
        #     print(i + 1, torch.max(depth))
        #     # os.system('pause')

        # Check for NaN values in gravity data
        if torch.isnan(torch.max(dg)):
            print(i+1, torch.max(depth))
        
        # Track maximum gravity anomaly value
        # if(torch.max(dg)>sample_max):
        #     sample_max = torch.max(dg)

# After inspection, found distortions in the training set at samples 20, 297, 600 causing NaN values in forward modeling
# Test set samples 871, 874, 883, 864, 865 show abnormal distortions
