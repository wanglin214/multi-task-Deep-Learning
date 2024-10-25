# 计算反演结果的均方根偏差，界面起伏相关系数
import numpy as np
from utils import readGrd
import os


def rmse(a, b):
    # a 和 b 是两个 numpy 数组，可以是一维或二维的
    # 检查 a 和 b 的形状是否一致，如果不一致，抛出异常
    if a.shape != b.shape:
        raise ValueError("The shapes of a and b must be the same.")
    # 计算 a 和 b 的差值的平方
    diff = (a - b) ** 2
    # 对差值的平方求平均，得到均方差
    mse = np.mean(diff)
    # 对均方差开方，得到均方根偏差
    rmsd = np.sqrt(mse)
    # 返回均方根偏差
    return rmsd


def corrcoef(a, b):
    # a 和 b 是两个一维或二维的 numpy 数组
    # 检查 a 和 b 的形状是否一致，如果不一致，抛出异常
    if a.shape != b.shape:
        raise ValueError("The shapes of a and b must be the same.")
    # 将 a 和 b 展平成一维数组
    a_flat = a.flatten()
    b_flat = b.flatten()
    # 将 a 和 b 放入一个列表中
    arrays = [a_flat, b_flat]
    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(arrays)
    # 取出相关系数矩阵的第一行第二列的元素，即 a 和 b 之间的相关系数
    corr = corr_matrix[0, 1]
    # 返回相关系数
    return corr


if __name__ == '__main__':
    ModNum = 512    # 测试集数量
    DensNum = 41  # 每个界面的密度步长数
    EvaNum = ModNum * DensNum  # 总的评估数量
    rmse_depth = np.zeros(EvaNum)
    err_density = np.zeros(EvaNum)
    rmse_dg = np.zeros(EvaNum)
    coef = np.zeros(EvaNum)
    testpath = r"D:\Project\DL_interface_inversion\data\Test"

    predmod_names = [i for i in os.listdir(os.path.join(testpath, "pred_model")) if i.endswith(".grd")]
    predmod_list = [os.path.join(testpath, "pred_model", i) for i in predmod_names]  # 获取dg文件夹下的全部grd文件
    total_err = []

    for i in range(len(predmod_list)):
        # 反演结果及异常文件名
        predmod_name = predmod_names[i]
        # 模型与密度编号
        imodel = predmod_name[-23:-18]
        idens = predmod_name[-12:-10]
        true_dens = 0.1 + (int(idens) - 1) * 0.6 / 40.0
        pred_dens = float(predmod_name[-9:-4])
        # print(imodel, idens, true_dens, pred_dens, type(pred_dens))
        # os.system('pause')
        dg_name = os.path.join(testpath,
                               'dg',
                               'FwGravofBasin_' + imodel + '_Dens_' + idens + '.grd')
        mod_name = os.path.join(testpath,
                                'model',
                                'SedofBasin_' + imodel + '.grd')
        preddg_name = os.path.join(testpath, 'pred_dg', predmod_name)
        # 读取数据
        dg = readGrd.readGrdbynp(dg_name)
        model = readGrd.readGrdbynp(mod_name)
        pred_model = readGrd.readGrdbynp(predmod_list[i])
        pred_dg = readGrd.readGrdbynp(preddg_name)

        # 计算深度，异常的均方差，密度的误差，深度相关系数
        rmse_depth[i] = rmse(pred_model, model)
        err_density[i] = abs(pred_dens - true_dens)
        rmse_dg[i] = rmse(pred_dg, dg)
        coef[i] = corrcoef(pred_model, model)
        total_err.append([(i + 1), np.max(model), rmse_depth[i],
                          err_density[i], rmse_dg[i], coef[i],
                          rmse_depth[i] / np.max(model)])

    # 统计结果
    depth_mean = np.mean(rmse_depth)
    depth_std = np.std(rmse_depth)
    density_mean = np.mean(err_density)
    density_std = np.std(err_density)
    dg_mean = np.mean(rmse_dg)
    dg_std = np.std(rmse_dg)
    coef_mean = np.mean(coef)
    coef_std = np.std(coef)

    print(f'depth_mean :{depth_mean}-depth_std===>>{depth_std}')
    print(f'density_mean :{density_mean}-density_std===>>{density_std}')
    print(f'dg_mean :{dg_mean}-dg_std===>>{dg_std}')
    print(f'coef_mean :{coef_mean}-depth_std===>>{coef_std}')
    print(f'coef_min :{np.min(coef)}-depth_max===>>{np.max(coef)}')
    np.savetxt('total_err_1e100_scale_smooth.txt', np.array(total_err))
