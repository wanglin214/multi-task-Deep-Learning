import os

import torch
from torch.utils.data import DataLoader
from utils import dataload
import tqdm


myroot = r"D:\Project\DL_interface_inversion\data"
batchsize = 1

train_loader = DataLoader(dataload.GravityDataset(myroot, train='test'),
                          batch_size=batchsize,
                          shuffle=False,
                          prefetch_factor=4,
                          num_workers=8,
                          )  # 加载训练数据

dataset_size = len(train_loader)
# print(dataset_size)
if __name__ == '__main__':
    sample_max = 0
    for i, (dg, depth, density) in enumerate(tqdm.tqdm(train_loader)):
        # if torch.max(depth)>17.0:
        #     print(i + 1, torch.max(depth))
        #     # os.system('pause')

        if torch.isnan(torch.max(dg)):
            print(i+1, torch.max(depth))
        # if(torch.max(dg)>sample_max):
        #     sample_max = torch.max(dg)
# 经过检查，训练集中的第20,297， 600界面有畸变，造成正演nan
# 测试集871,874,883，864,865 异常畸变