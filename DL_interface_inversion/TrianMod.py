# 导入torch模块
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler  # 混合精度所需要的模块
# 导入数据加载及网络模型模块
from utils import dataload, MinMax_Scaler
from netmodel import HybirdNet64
# 导入系统、时间及numpy
import os
import time
import numpy as np
# 导入可视化美化模块
import tqdm
from visdom import Visdom


def worker_init_fn(worker_id):
    time.sleep(worker_id * 0.02)  # 增加启动器间隔，可根据实际情况调整


# 在硬件设备（CPU、GPU）不同时，完全的可复现性无法保证，即使随机种子相同。
# 但是，在同一个设备上，应该保证可复现性。具体法是，在程序开始的时候固定torch的随机种子，同时也把numpy的随机种子固定
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()  #不建议使用torch.cuda.empty_cache()
# # 这个命令并不会真正地帮助你清理更多的显存，与此同时，还会让你的代码速度变慢
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
                          worker_init_fn=worker_init_fn)  # 加载训练数据，num_workers启动时间越久

test_loader = DataLoader(dataload.GravityDataset(myroot, train='validation'),
                         batch_size=batchsize // 2,
                         shuffle=True,
                         pin_memory=True,
                         drop_last=True,
                         prefetch_factor=64,
                         num_workers=2,
                         worker_init_fn=worker_init_fn)  # 加载测试数据

net = HybirdNet64.multitask()  # 网络定义为我自定义的multitask
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
    opt = optim.AdamW(net.parameters(), lr=base_lr)  # 定义最优化方法为Adam，可选参数需要调整
    # # 余弦退火学习率衰减策略
    # scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6, last_epoch=-1)
    # 初始化学习率调度器（ReduceLROnPlateau）
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=True)
    # 定义损失函数
    loss_func = nn.MSELoss().to(device)  # 定义损失函数为平方损失

    # 定义总的损失函数监控对象
    viz_train = Visdom()  # 创建Visdom实例：在conda prompt使用 python -m visdom.server命令
    viz_train.line([0.], [0.], win='total_loss', opts=dict(title='train_loss_total'))
    viz_validation = Visdom()  # 创建Visdom实例
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
                density.to(device, non_blocking=True)  # 将输入输出数据加载到计算设备上
            # 数据归一化，增加稳定性
            dg_scaled = MinMax_Scaler.min_max_normalize(dg, dg_min, dg_max)
            depth_scaled = MinMax_Scaler.min_max_normalize(depth, depth_min, depth_max)
            density_scaled = MinMax_Scaler.min_max_normalize(density, dens_min, dens_max)

            opt.zero_grad()  # 3. 梯度清零反向传播
            # 训练使用混合精度
            with autocast():
                pred_depth, pred_density = net(dg_scaled)
                # 2. 计算损失(网络输出反归一化之后传入损失函数计算）
                l1 = loss_func(pred_depth, depth_scaled)  # 每个batch的loss
                l2 = loss_func(pred_density, density_scaled)
                loss_total = l1 + l2  # + trainloss_gravity * 0.2
                trainloss_depth = trainloss_depth + l1
                trainloss_density = trainloss_density + l2
                trainloss_total = trainloss_total + loss_total

            scaler.scale(loss_total).backward()
            scaler.step(opt)
            scaler.update()

            # if (i + 1) % 200 == 0:  # 40个batch输出一次参数
            #     print(f'{epoch + 1}-{i + 1}-trainloss_depth===>>{l1.item() }')
            #     print(f'{epoch + 1}-{i + 1}-trainloss_density==  =>>{l2.item() }')
            #     # print(f'{epoch + 1}-{i + 1}-trainloss_gravity===>>{trainloss_gravity.item()}')
            #     print(f'{epoch + 1}-{i + 1}-trainloss_total===>>{loss_total.item()}')

        # 每个epoch计算一次平均损失输出，len(train_loader)=18000/64= 281，最后一次不满64的drop
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
        # train_log_total.append(trainloss_total.item())  每个epoch输出损失函数visdom占用时间影响GPU利用率
        #  考虑每个epoch计算一个平均损失函数
        viz_train.line([trainloss_total.item()], [epoch + 1], win='train_loss', update='append')

        if epoch == 0:
            start = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
            print(start)
        # # 使用余弦退火学习率衰减策略
        # scheduler.step()
        epoch = epoch + 1
        if epoch % 10 == 0:
            checkpoint = {"model_state_dict": net.state_dict(),
                          "optimizer_state_dict": opt.state_dict(),
                          "epoch": epoch}
            path_checkpoint = "./params/checkpoint_multitask_{}_1e100_40km_scale_smmoth.pth".format(epoch)
            torch.save(checkpoint, path_checkpoint)  # 每隔5个epoch保存一个断点文件
            print(' weight save successfully!')

        net.eval()  # 进行验证不需要梯度
        # 一个epoch束做一次validation
        with torch.no_grad():  # validation过程不需要梯度信息，关闭梯度再进行，降低显存占用
            testloss_depth = 0
            testloss_density = 0
            for dg_validation, depth_validation, density_validation in test_loader:
                dg_validation, depth_validation, density_validation = dg_validation.to(device), \
                    depth_validation.to(device), density_validation.to(device)  # 将输入输出数据加载到计算设备上

                # 数据归一化，增加稳定性
                dg_validation_scaled = MinMax_Scaler.min_max_normalize(dg_validation, dg_min, dg_max)
                depth_validation_scaled = MinMax_Scaler.min_max_normalize(depth_validation, depth_min, depth_max)
                density_validation_scaled = MinMax_Scaler.min_max_normalize(density_validation, dens_min, dens_max)

                with autocast():
                    pred_depth_val, pred_density_val = net(dg_validation_scaled)
                    # 2. 计算损失(网络输出反归一化之后传入损失函数计算）
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

        # 记录当前学习率，并根据验证集损失函数不下降时自动调整学习率
        lr_step.append([epoch, opt.param_groups[0]["lr"]])
        scheduler.step(testloss_total)

    train_log_depth = np.array(train_log_depth)
    validation_log_depth = np.array(validation_log_depth)
    train_log_density = np.array(train_log_density)
    validation_log_density = np.array(validation_log_density)
    # train_log_gravity = np.array(train_log_gravity)
    #     # validation_log_gravity = np.array(validation_log_gravity)
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
    print('finished training!')  # 所有epoch训练完毕
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    torch.save(net.state_dict(), weight_path)  # 保存为最终版本权值
