import torch
from torch.utils.data import DataLoader
from utils import dataload
import tqdm

myroot = r"D:\Project\DL_interface_inversion\data"  # Root directory for data
batchsize = 2056  # Batch size for data loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device for computation

# Create DataLoader for training data
train_loader = DataLoader(dataload.GravityDataset(myroot, train='train'),
                          batch_size=batchsize,
                          shuffle=False,
                          prefetch_factor=4,
                          num_workers=8,
                          )  # Load training data

dataset_size = len(train_loader)
print(dataset_size)
dg_max = 0.0  # Initialize maximum gravity anomaly value
depth_max = 0.0  # Initialize maximum depth value

if __name__ == '__main__':
    sample_max = 0
    # Iterate through the dataset to find maximum values
    for i, (dg, depth, density) in enumerate(tqdm.tqdm(train_loader)):
        # Move data to device (GPU if available)
        dg, depth, density = dg.to(device, non_blocking=True), depth.to(device, non_blocking=True), \
            density.to(device, non_blocking=True)  # Load input and output data to computation device

        # Update maximum gravity anomaly value if needed
        if torch.max(dg) > dg_max:
            dg_max = torch.max(dg)

        # Update maximum depth value if needed
        if torch.max(depth) > depth_max:
            depth_max = torch.max(depth)

    print(dg_max, depth_max)
    # Maximum values found in different datasets:
    # test: tensor(436.4686, device='cuda:0') tensor(16.0004, device='cuda:0')
    # validation: tensor(407.4593, device='cuda:0') tensor(16.0003, device='cuda:0')
    # train: tensor(445.6873, device='cuda:0') tensor(16.2861, device='cuda:0')
    # Normalized ranges for parameters:
    # dg_max=500mGal, dg_min=0.0; dpeth_max= 17km, depth_min=0.0; density_max=0.7,density_min=0.0
