import os
import numpy as np
import torch
from utils import readGrd, outGrd
from netmodel import HybirdNet64

from utils import MinMax_Scaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

weight_path = 'params\HybirdNet64_1e100_40km_scale_smmoth.pth'
# path_checkpoint =  'params/checkpoint_multitask_200_400.pth''[
myroot = r"D:\Project\DL_interface_inversion\data\Yucca Flat\grav_YuccaFlat_1660m_d64.grd"
out_path = r"D:\Project\DL_interface_inversion\data\Yucca Flat"

if __name__ == '__main__':  # Load pre-trained weights for prediction
    net = HybirdNet64.multitask()  # Define network as my custom multitask model
    if os.path.exists(weight_path):
        # checkpoint = torch.load(path_checkpoint)
        # net.load_state_dict(checkpoint['model_state_dict'])
        net.load_state_dict(torch.load(weight_path, map_location=device))
        net = net.to(device)
        net.eval()  # Enable evaluation/inference mode, disable gradient backpropagation for training
        net.zero_grad()

        print('successful load weight！')
    else:
        print('not successful load weight')

    dg_max = 500.0
    dg_min = 0.0
    depth_max = 17.0
    depth_min = 0.0
    dens_max = 0.7
    dens_min = 0.1

    dg = readGrd.readGrdbynp(myroot)
    dg_real = torch.from_numpy(dg).unsqueeze(0).unsqueeze(0)
    dg_real = (-1.0) * dg_real.float().to(device)
    dg_scaled = MinMax_Scaler.min_max_normalize(dg_real, dg_min, dg_max)
    # print(device, dg_real.device)
    # os.system('pause')
    pred_depth, pred_density = net(dg_scaled)
    out_model_test = MinMax_Scaler.min_max_inverse(pred_depth, depth_min, depth_max)  # .unsqueeze(0)
    out_model_test = out_model_test.squeeze(0).squeeze(0).cpu()  # Reinterpolate to original data size and convert to numpy array
    out_model_test = out_model_test.detach().numpy()
    pred_dens = MinMax_Scaler.min_max_inverse(pred_density, dens_min, dens_max).item()  # Convert numpy array of size 1 to float
    # pred_dens = pred_density.item()
    # print(pred_dens)
    out_dens_str = "{:.3f}".format(pred_dens)
    filepath = os.path.join(out_path, 'YuccaPreDepth64_Dens_smooth_scale_1e100_' + out_dens_str + '.grd')
    outGrd.outGrd(filepath, out_model_test, np.min(out_model_test), np.max(out_model_test))
    print(np.max(out_model_test), out_dens_str)
