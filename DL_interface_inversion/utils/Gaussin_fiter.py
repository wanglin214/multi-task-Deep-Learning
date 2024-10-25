import os
import numpy as np
from scipy.ndimage import gaussian_filter
from utils import readGrd, outGrd

# 导入 matplotlib 库
import matplotlib.pyplot as plt

sigma = 4
myroot = r"D:\Project\DL_interface_inversion\data\field\bishop64x_basement_km.grd"
filepath = r"D:\Project\DL_interface_inversion\data\field\bishop64x_basement_km_filter.grd"

if __name__ == '__main__':  # 加载已经训练好的训练权值进行预测

    dg = readGrd.readGrdbynp(myroot)

    # 调用 gaussian_filter 函数，对数据进行高斯滤波
    filtered_data = gaussian_filter(dg, sigma)


    # 定义等值线的层数和颜色
    levels = 10
    cmap = plt.cm.coolwarm

    # 创建画布和子图
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制原始数据的等值线图
    ax[0].contourf(dg, levels, cmap=cmap)
    ax[0].set_title("Original data")

    # 绘制滤波后数据的带填充的等值线图
    cs = ax[1].contourf(filtered_data, levels, cmap=cmap)
    ax[1].set_title("Filtered data")

    # 在画布上添加以 cs 中绘图数据为参考的色标
    fig.colorbar(cs, shrink=0.8)

    # 显示图形
    plt.show()

    outGrd.outGrd(filepath, filtered_data, np.min(filtered_data), np.max(filtered_data))