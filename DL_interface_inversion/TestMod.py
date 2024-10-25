import os
import numpy as np
# import tqdm
import torch.nn.functional as F
from torch import nn
# from torch.optim import lr_scheduler
import torch
from torch.utils.data import DataLoader
from utils import dataload, outGrd, MinMax_Scaler
from netmodel import HybirdNet64

from visdom import Visdom

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()  #不建议使用torch.cuda.empty_cache()
# # 这个命令并不会真正地帮助你清理更多的显存，与此同时，还会让你的代码速度变慢
weight_path = 'params\HybirdNet64_1e100_40km_scale_smmoth.pth'
# path_checkpoint = 'params/checkpoint_multitask_100_64e100_40km_scale_smmoth.pth'
myroot = r"D:\Project\DL_interface_inversion\data"
batchsize = 1

test_loader = DataLoader(dataload.GravityDataset(myroot, train='test'),
                         batch_size=batchsize,
                         shuffle=False,
                         pin_memory=True,
                         prefetch_factor=2,
                         num_workers=2)  # 加载测试数据
# print(test_loader.dataset.dg_list[0:20])
# os.system('pause')

net = HybirdNet64.multitask()  # 网络定义为我自定义的multitask
net = net.to(device)
net.eval()  # 打开检验预测推理模式，关闭梯度逆向传播训练
net.zero_grad()

if __name__ == '__main__':  # 加载已经训练好的训练权值进行预测

    dg_max = 500.0
    dg_min = 0.0
    depth_max = 17.0
    depth_min = 0.0
    dens_max = 0.7
    dens_min = 0.1

    if os.path.exists(weight_path):
        # checkpoint = torch.load(path_checkpoint)
        # net.load_state_dict(checkpoint['model_state_dict'])
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    viz_3 = Visdom()  # 创建Visdom实例
    viz_3.line([0.], [0.], win='validation_errors', opts=dict(title='val_err_total'))

    loss_fun = nn.MSELoss().to(device)  # 定义损失函数为平方损失
    test_loss = 0

    testloss_depth = 0
    testloss_density = 0
    # testloss_total = 0
    test_num = 0
    test_log_depth = []
    test_log_density = []
    test_log_total = []

    for dg_test, depth_test, density_test in test_loader:
        dg_test, depth_test, density_test = dg_test.to(device), depth_test.to(device), density_test.to(
            device)  # 将输入输出数据加载到计算设备上
        # print(dg_test.shape,type(dg_test))
        # os.system('pause')
        # 数据归一化，增加稳定性
        dg_scaled = MinMax_Scaler.min_max_normalize(dg_test, dg_min, dg_max)
        # depth_scaled = MinMax_Scaler.min_max_normalize(depth_test, depth_min, depth_max)
        # density_scaled = MinMax_Scaler.min_max_normalize(density_test, dens_min, dens_max)

        pred_depth, pred_density = net(dg_scaled)
        # 2. 计算损失(网络输出反归一化之后传入损失函数计算）
        # 预测结果反归一化计算真实误差
        pred_depth_inv = MinMax_Scaler.min_max_inverse(pred_depth, depth_min, depth_max)
        pred_density_inv = MinMax_Scaler.min_max_inverse(pred_density, dens_min, dens_max)

        testloss_depth = torch.sqrt(loss_fun(pred_depth_inv, depth_test))
        testloss_density = torch.sqrt(loss_fun(pred_density_inv, density_test))
        testloss_total = testloss_depth + testloss_density
        if (test_num + 1) % 10 == 0:  # 10个batch输出一次参数
            print(f'{test_num + 1}-testloss_depth===>>{testloss_depth.item()}')
            print(f'{test_num + 1}-testloss_density===>>{testloss_density.item()}')
            # print(f'{epoch + 1}-{i + 1}-trainloss_gravity===>>{trainloss_gravity.item()}')
            print(f'{test_num + 1}-testloss_total===>>{testloss_total.item()}')
        # err记录真实误差，进行平均误差，最大最小误差，均方误差统计
        # err = out_model_test - mod_label_test
        test_num = test_num + 1
        # viz_1.line([testloss_depth.item()], [test_num], win='test_loss', update='append')
        # viz_2.line([testloss_density.item()], [test_num], win='test_loss', update='append')
        viz_3.line([testloss_total.item()], [test_num], win='test_loss', update='append')
        # 输出预测结果, 重插值为 200*500*500
        out_model_test = pred_depth_inv  # .unsqueeze(0)
        pred_dens = float(pred_density_inv.cpu().detach().numpy())  # 将size为1的numpy数组转为浮点数
        # print(pred_dens)
        out_dens_str = "{:.3f}".format(pred_dens)
        # print(out_dens_str)
        # os.system('pause')
        # out_model_test = F.interpolate(out_model_test, size=[100, 100], mode='bilinear', align_corners=True)
        out_model_test = out_model_test.squeeze(0).squeeze(0).cpu()  # 重新插值为原始数据尺寸并转为numpy数组
        out_model_test = out_model_test.detach().numpy()
        # print(pred_densMod.shape)
        filename = test_loader.dataset.dg_list[test_num - 1][-23:-4]
        # print(filename)
        filepath = os.path.join(myroot, 'Test', 'pred_model', filename + '_' + out_dens_str + '.grd')
        # print(filepath)
        # os.system('pause')
        outGrd.outGrd(filepath, out_model_test, np.min(out_model_test), np.max(out_model_test))
        # print(filepath, filename)
        test_log_depth.append([test_num, torch.max(depth_test).item(), density_test.item(), testloss_depth.item()])
        test_log_density.append([test_num, torch.max(depth_test).item(), density_test.item(), testloss_density.item()])
        test_log_total.append([test_num, torch.max(depth_test).item(), density_test.item(), testloss_total.item()])

        # print(f'the-{test_num}-validation-MSE-error- is===> {test_loss.item()}')

    test_log_depth = np.array(test_log_depth)
    test_log_density = np.array(test_log_density)
    test_log_total = np.array(test_log_total)
    # print(type(test_log), test_log.shape)
    # os.system('pause')
    np.savetxt('test_log_depth_1e100_40km_scale_smmoth.txt', test_log_depth)
    np.savetxt('test_log_density_1e100_40km_scale_smmoth.txt', test_log_density)
    np.savetxt('test_log_total_1e100_40km_scale_smmoth.txt', test_log_total)
