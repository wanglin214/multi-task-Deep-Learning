# 批量重命名文件，只需要在脚本中批量设置重命名后的文件名即可
import os

# 用法示例：在目录中的所有 .grd 文件前加上前缀 "new_"，后缀 "_data"，并将它们移动到新目录
inpath = r"E:\F盘学习资料\frequency_interface_forward\FmtGrd"
outpath = r"E:\F盘学习资料\frequency_interface_forward\Basement"
# 获取输入目录下的所有文件
files = os.listdir(inpath)
# print(files.__len__(), type(files))
Nfiles = files.__len__()

i = 0

for filename in files:
    # 构建完整的输入文件路径
    old_path = os.path.join(inpath, filename)
    i = i+1
    # 检查文件是否是文件而不是目录
    if os.path.isfile(old_path):
        # 构建新的文件名
        new_filename = 'SedofBasin_' + "{:05d}".format(i) + '.grd'
        # 构建完整的输出文件路径
        new_path = os.path.join(outpath, new_filename)
        # 执行重命名和移动
        os.rename(old_path, new_path)
        print(f'Renamed and moved: {filename} -> {new_filename} to {outpath}')


