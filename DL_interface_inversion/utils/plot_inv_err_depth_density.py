import pandas as pd
import matplotlib.pyplot as plt

# 假设数据保存在一个 CSV 文件中，以空格分隔
errpath = r'D:\Project\DL_interface_inversion\utils\total_err_1e100_scale.txt'
# 如果数据是以其他方式分隔的，可以修改 sep 参数
df = pd.read_csv(errpath, sep=' ')

# 选择需要的列，这里选择前10列
selected_columns = df.iloc[:, 6]

# 将选定列的值展平成一个 Series
flat_data = selected_columns.values

# 绘制直方图
plt.hist(flat_data, bins=20, edgecolor='black',density=True)
# plt.xlabel('RMSE of depth inversion/m')
# plt.xlabel('ABSE of density inversion/(g/cm³)')
# plt.xlabel('RMSE of reconstructed gravity anomaly/mGal')
# plt.xlabel('Morphology Similarity')
plt.xlabel('Relative RMSE of depth inversion')

plt.ylabel('Frequency')
# plt.title('Histogram')
# plt.xlim(-1000, 1000)
plt.show()
