import pandas as pd
import matplotlib.pyplot as plt

# 假设数据保存在一个 CSV 文件中，以空格分隔
errpath = r'D:\Project\DL_interface_inversion\data\Yucca Flat\err_PreDepth64_Dens_scale_1e100_0.587.dat'
# 如果数据是以其他方式分隔的，可以修改 sep 参数
df = pd.read_csv(errpath, sep=' ')

# 选择需要的列，这里选择前10列
selected_columns = df.iloc[:, 10]

# 将选定列的值展平成一个 Series
flat_data = selected_columns.values

# 绘制直方图
plt.hist(flat_data*1000.0, bins=20, edgecolor='black',density=True)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.xlim(-1000, 1000)

# 计算平均值和标准差
mean_value = flat_data.mean()
std_dev = flat_data.std()

# 在直方图上显示统计平均值和标准差
plt.axvline(mean_value*1000.0, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_value:.2f}')
plt.axvline((mean_value + std_dev)*1000.0, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev = {std_dev:.2f}')
plt.axvline((mean_value - std_dev)*1000.0, color='green', linestyle='dashed', linewidth=2)

# 显示图例
plt.legend()

# 在图中显示文字
plt.text(-950, 0.007, f'Mean = {mean_value:.2f}', color='red')
plt.text(-950, 0.006, f'Std Dev = {std_dev:.2f}', color='green')

plt.show()
plt.show()