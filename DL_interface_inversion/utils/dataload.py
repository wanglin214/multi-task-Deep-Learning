import os

import torch
# import numpy as np
# import torch.nn.functional as fun
from torch.utils.data import Dataset  # 构造数据集，支持索引，总长度
# from torch.utils.data import DataLoader

from utils.readGrd import readGrdbynp

"""
深度学习训练框架/步骤：
   1. prepare dataset
      tools: Dataset and DataLoader
   2. Design model using Class
      inherit from nn.moudle
   3. Construct loss and optimizer
      using Pytorch API
   4. Training cycle
   forward, backward, update
"""


class GravityDataset(Dataset):

    def __init__(self, root: str, train: str):  # root指定根目录,resize自定义数据大小
        super(GravityDataset, self).__init__()

        self.train = train
        if self.train == 'train':
            self.root = os.path.join(root, 'Train')
        elif self.train == 'test':
            self.root = os.path.join(root, 'Test')
        elif self.train == 'validation':
            self.root = os.path.join(root, 'Validation')

        assert os.path.join(self.root), f"path '{self.root}' does not exists."  # 判断文件路径是否存在

        dg_names = [i for i in os.listdir(os.path.join(self.root, "dg")) if i.endswith(".grd")]
        self.dg_list = [os.path.join(self.root, "dg", i) for i in dg_names]  # 获取dg文件夹下的全部grd文件
        self.interface_list = [str(i) for i in range(len(self.dg_list))]
        # self.density = [i for i in range(len(self.dg_list))]
        for i in range(len(self.dg_list)):  # 让异常与对应的模型label数据匹配
            censtr = self.dg_list[i][-17:-12]  # 取出文件中间编号部分
            # print(censtr)
            # os.system('pause')
            self.interface_list[i] = os.path.join(self.root, 'model', 'SedofBasin_' + censtr + '.grd')

        # check files
        for i in self.dg_list:  # 注意for i in self.dg_list或者dg_names中i为字符串类型而非整形
            if os.path.exists(i) is False:
                raise FileExistsError(f"file {i} doesn't exissts.")

    def __getitem__(self, index):  # 按照文件列表当前索引下标对应的sample与label
        dg = readGrdbynp(self.dg_list[index])
        label_interface = readGrdbynp(self.interface_list[index])
        # 获取文件名中密度编号
        numdens = self.dg_list[index][-6:-4]
        label_dens = 0.1 + (int(numdens) - 1) * (0.7 - 0.1) / 40.0
        label_dens = torch.tensor(label_dens)
        # print(numdens, label_dens)
        # os.system('pause')
        # dg = (dg - np.mean(dg)) / np.std(dg)   #对异常进行标准化操作,使得数据满足均值为0，方差为1
        # dg = (dg - np.min(dg)) / (np.max(dg)-np.min(dg))  # 最大最小归一化[0,1]
        # numpy数组传入torch
        label_interface = torch.from_numpy(label_interface).unsqueeze(0)
        dg = torch.from_numpy(dg).unsqueeze(0)
        # print(mod_label.shape,dg.shape)
        dg = dg * (-1.0)
        mod_label = label_interface  # 设置输入输出为单精度float16 类型Tensor
        # # 对数据重采样重构成网络目标输入大小200*200, 200*200，二维数据interpolate函数输入需为[b,c,h,w]
        # dg_sample = fun.interpolate(dg, size=[64, 64], mode='bilinear', align_corners=True)
        # lab_interface_sample = fun.interpolate(mod_label, size=[64, 64], mode='bilinear', align_corners=True)
        #  将batch通道数维度去掉
        dg_sample = dg
        label_interface_sample = mod_label
        label_dens = label_dens.unsqueeze(0)
        # print(dg_sample.shape,mod_label_sample.shape)
        # print(scaler)
        return dg_sample.float(), label_interface_sample.float(), label_dens.float()

    def __len__(self):
        return len(self.dg_list)


# # # 测试数据集读取是否成功
if __name__ == '__main__':
    myroot = r"D:\Project\DL_interface_inversion\data"
    mydata = GravityDataset(myroot, train='train')
    print(mydata.__len__())
    print(mydata[40][0].shape, mydata[40][1].shape, mydata[40][2].shape)
    print(torch.max(mydata[40][0]))
