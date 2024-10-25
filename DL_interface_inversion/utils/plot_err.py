import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
file_path = r'D:\Project\DL_interface_inversion\test_log_depth_e200_40km.txt'  # 替换为你的文件路径
data = pd.read_csv(file_path, sep='\s+', header=None)

# 分离数据
x = data.iloc[:, 0]
y = data.iloc[:, 3]

# 绘制柱状图
plt.bar(x, y, color='blue', alpha=0.7)

# 添加坐标轴标签
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 添加标题
plt.title('Bar Chart of Column 1 and Column 3')
# 显示图形
plt.show()