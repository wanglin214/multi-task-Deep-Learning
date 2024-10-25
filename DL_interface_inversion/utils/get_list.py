# 计算反演结果的均方根偏差，界面起伏相关系数
import numpy as np
import os

if __name__ == '__main__':
    testpath = r"D:\Project\DL_interface_inversion\data\Test"
    predmod_names = [i for i in os.listdir(os.path.join(testpath, "pred_model")) if i.endswith(".grd")]
    predmod_list = [os.path.join(testpath, "pred_model", i) for i in predmod_names]  # 获取dg文件夹下的全部grd文件
    dens = np.zeros(len(predmod_list))
    for i in range(len(predmod_list)):
        dens[i] = float(predmod_names[i][-9:-4])
        # print(dens[i])
        # os.system('pause')

    print(len(predmod_list),predmod_list[0])
    with open("dir_grd_40km_scale_smooth.txt", "w") as file:
        # 将字符串列表写入文件，每个字符串占一行
        for i in range(len(predmod_list)):
            file.write(predmod_names[i] +' ' + str(dens[i]) + "\n")

    print("字符串列表已保存到 output.txt 文件中。")

