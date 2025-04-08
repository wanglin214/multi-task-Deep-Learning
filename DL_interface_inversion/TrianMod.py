# Import torch modules
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler  # Modules needed for mixed precision training
# Import data loading and network model modules
from utils import dataload, MinMax_Scaler
from netmodel import HybirdNet64
# Import system, time and numpy
import os
import time
import numpy as np
# Import visualization modules
import tqdm
from visdom import Visdom


def worker_init_fn(worker_id):
    time.sleep(worker_id * 0.02)  # Add interval between worker startups, can be adjusted based on actual situation


# Complete reproducibility cannot be guaranteed across different hardware devices (CPU, GPU), even with the same random seed.
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()  # Not recommended to use torch.cuda.empty_cache()

weight_path = r'D:\Project\DL_interface_inversion\params\HybirdNet64_1e100_40km_scale_smmoth.pth'
myroot = r"D:\Project\DL_interface_inversion\data"
batchsize = 256

train_loader = DataLoader(dataload.GravityDataset(myroot, train='train'),
                          batch_size=batchsize,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True,
                          prefetch_factor=128,
                          num_workers=2,
                          worker_init_fn=worker_init_fn)  # Load training data, num_workers takes longer to start with higher values

test_loader = DataLoader(dataload.GravityDataset(myroot, train='validation'),
                         batch_size=batchsize // 2,
                         shuffle=True,
                         pin_memory=True,
                         drop_last=True,
                         prefetch_factor=64,
                         num_workers=2,
                         worker_init_fn=worker_init_fn)  # Load test data

net = HybirdNet64.multitask()  # Define network as my custom multitask model
net = net.to(device)
# print(len(test_loader))
# os.system('pause')

if __name__ == '__main__':
    if os.path.exists(weight_path):
        print('weight_path exists')
        # net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('weight_path not exists')

    dg_max = 500.0
    dg_min = 0.0
    depth_max = 17.0
    depth_min = 0.0
    dens_max = 0.7
    dens_min = 0.1

    base_lr = 1e-3
    scaler = GradScaler()
    opt = optim.AdamW(net.parameters(), lr=base_lr)  # Define optimization method as Adam, optional parameters need adjustment
    # # Cosine annealing learning rate decay strategy
    # scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6, last_epoch=-1)
    # Initialize learning rate scheduler (ReduceLROnPlateau)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=True)
    # Define loss function
    loss_func = nn.MSELoss().to(device)  # Define loss function as mean squared error

    # Define total loss function monitoring objects
    viz_train = Visdom()  # Create Visdom instance: use 'python -m visdom.server' command in conda prompt
    viz_train.line([0.], [0.], win='total_loss', opts=dict(title='train_loss_total'))
    viz_validation = Visdom()  # Create Visdom instance
    viz_validation.line([0.], [0.], win='total_loss', opts=dict(title='validation_loss_total'))

    epoch = 0
    train_log_depth = []
    validation_log_depth = []
    train_log_density = []
    validation_log_density = []
    # train_log_gravity = []
    # validation_log_gravity = []
    train_log_total = []
    validation_log_total = []
    lr_step = []

    while epoch < 100:
        net.train()
        # adjust_lr(epoch + 1)
        trainloss_depth = 0.0
        trainloss_density = 0.0
        trainloss_total = 0.0
        for i, (dg, depth, density) in enumerate(tqdm.tqdm(train_loader)):
            dg, depth, density = dg.to(device, non_blocking=True), depth.to(device, non_blocking=True), \
                density.to(device, non_blocking=True)  # Load input and output data to computing device
            # Data normalization to increase stability
            dg_scaled = MinMax_Scaler.min_max_normalize(dg, dg_min, dg_max)
            depth_scaled = MinMax_Scaler.min_max_normalize(depth, depth_min, depth_max)
            density_scaled = MinMax_Scaler.min_max_normalize(density, dens_min, dens_max)

            opt.zero_grad()  # 3. Zero gradients for backpropagation
            # Use mixed precision for training
            with autocast():
                pred_depth, pred_density = net(dg_scaled)
                # 2. Calculate loss (network output is denormalized before being passed to loss function)
                l1 = loss_func(pred_depth, depth_scaled)  # Loss for each batch
                l2 = loss_func(pred_density, density_scaled)
                loss_total = l1 + l2  # + trainloss_gravity * 0.2
                trainloss_depth = trainloss_depth + l1
                trainloss_density = trainloss_density + l2
                trainloss_total = trainloss_total + loss_total

            scaler.scale(loss_total).backward()
            scaler.step(opt)
            scaler.update()

            # if (i + 1) % 200 == 0:  # Output parameters every 40 batches
            #     print(f'{epoch + 1}-{i + 1}-trainloss_depth===>>{l1.item() }')
            #     print(f'{epoch + 1}-{i + 1}-trainloss_density==  =>>{l2.item() }')
            #     # print(f'{epoch + 1}-{i + 1}-trainloss_gravity===>>{trainloss_gravity.item()}')
            #     print(f'{epoch + 1}-{i + 1}-trainloss_total===>>{loss_total.item()}')

        # Calculate average loss for each epoch, len(train_loader)=18000/64=281, last batch dropped if not full 64
        trainloss_depth = trainloss_depth / len(train_loader)
        trainloss_density = trainloss_density / len(train_loader)
        trainloss_total = trainloss_total / len(train_loader)

        print(f'epoch:{epoch + 1}-trainloss_depth===>>{trainloss_depth.item()}')
        print(f'epoch:{epoch + 1}-trainloss_density===>>{trainloss_density.item()}')
        # print(f'epoch:{epoch}-testloss_gravity===>>{testloss_gravity.item()}')
        print(f'epoch:{epoch + 1}-trainloss_total===>>{trainloss_total.item()}')

        train_log_depth.append([epoch + 1, trainloss_depth.item()])
        train_log_density.append([epoch + 1, trainloss_density.item()])
        train_log_total.append([epoch + 1, trainloss_total.item()])
        # train_log_total.append(trainloss_total.item())  # Visdom output of loss function for each epoch affects GPU utilization
        # Consider calculating an average loss function for each epoch
        viz_train.line([trainloss_total.item()], [epoch + 1], win='train_loss', update='append')

        if epoch == 0:
            start = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
            print(start)
        # # Use cosine annealing learning rate decay strategy
        # scheduler.step()
        epoch = epoch + 1
        if epoch % 10 == 0:
            checkpoint = {"model_state_dict": net.state_dict(),
                          "optimizer_state_dict": opt.state_dict(),
                          "epoch": epoch}
            path_checkpoint = "./params/checkpoint_multitask_{}_1e100_40km_scale_smmoth.pth".format(epoch)
            torch.save(checkpoint, path_checkpoint)  # Save a checkpoint file every 10 epochs
            print(' weight save successfully!')

        net.eval()  # Set to evaluation mode for validation (no gradients needed)
        # Perform validation once per epoch
        with torch.no_grad():  # Validation process doesn't need gradient information, disable gradients to reduce memory usage
            testloss_depth = 0
            testloss_density = 0
            for dg_validation, depth_validation, density_validation in test_loader:
                dg_validation, depth_validation, density_validation = dg_validation.to(device), \
                    depth_validation.to(device), density_validation.to(device)  # Load input and output data to computing device

                # Data normalization to increase stability
                dg_validation_scaled = MinMax_Scaler.min_max_normalize(dg_validation, dg_min, dg_max)
                depth_validation_scaled = MinMax_Scaler.min_max_normalize(depth_validation, depth_min, depth_max)
                density_validation_scaled = MinMax_Scaler.min_max_normalize(density_validation, dens_min, dens_max)

                with autocast():
                    pred_depth_val, pred_density_val = net(dg_validation_scaled)
                    # 2. Calculate loss (network output is denormalized before being passed to loss function)
                    testloss_depth = testloss_depth + loss_func(pred_depth_val, depth_validation_scaled)
                    testloss_density = testloss_density + loss_func(pred_density_val, density_validation_scaled)

            testloss_depth = testloss_depth / len(test_loader)
            testloss_density = testloss_density / len(test_loader)
            testloss_total = testloss_depth + testloss_density

            validation_log_depth.append([epoch, testloss_depth.item()])
            validation_log_density.append([epoch, testloss_density.item()])
            validation_log_total.append([epoch, testloss_total.item()])
            viz_validation.line([testloss_total.item()], [epoch], win='test_loss', update='append')

            print(f'epoch:{epoch}-testloss_depth===>>{testloss_depth.item()}')
            print(f'epoch:{epoch}-testloss_density===>>{testloss_density.item()}')
            # print(f'epoch:{epoch}-testloss_gravity===>>{testloss_gravity.item()}')
            print(f'epoch:{epoch}-testloss_total===>>{testloss_total.item()}')

        # Record current learning rate and automatically adjust learning rate when validation loss doesn't decrease
        lr_step.append([epoch, opt.param_groups[0]["lr"]])
        scheduler.step(testloss_total)

    train_log_depth = np.array(train_log_depth)
    validation_log_depth = np.array(validation_log_depth)
    train_log_density = np.array(train_log_density)
    validation_log_density = np.array(validation_log_density)
    # train_log_gravity = np.array(train_log_gravity)
    # validation_log_gravity = np.array(validation_log_gravity)
    train_log_total = np.array(train_log_total)
    validation_log_total = np.array(validation_log_total)
    lr_step = np.array(lr_step)

    np.savetxt('train_log_depth_1e100_40km_scale_smmoth.txt', train_log_depth)
    np.savetxt('validation_log_depth_1e100_40km_scale_smmoth.txt', validation_log_depth)
    np.savetxt('train_log_density_1e100_40km_scale_smmoth.txt', train_log_density)
    np.savetxt('validation_log_density_1e100_40km_scale_smmoth.txt', validation_log_density)
    np.savetxt('train_log_total_1e100_40km_scale_smmoth.txt', train_log_total)
    np.savetxt('validation_log_total_1e100_40km_scale_smmoth.txt', validation_log_total)
    np.savetxt('lr_step_1e100_40km_scale_smmoth.txt', lr_step)
    print('finished training!')  # All epochs training completed
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    torch.save(net.state_dict(), weight_path)  # Save as final version weights
