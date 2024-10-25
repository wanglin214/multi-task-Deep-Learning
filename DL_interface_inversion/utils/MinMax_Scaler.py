import torch


# 定义最大最小归一化的函数
def min_max_normalize(x, x_min, x_max):
    # 使用公式进行归一化
    x_norm = (x - x_min) / (x_max - x_min)

    # 返回归一化后的数据和归一化参数
    return x_norm


# 定义反归一化的函数
def min_max_inverse(x_norm, x_min, x_max):
    # 使用公式进行反归一化
    x = x_norm * (x_max - x_min) + x_min

    # 返回反归一化后的数据
    return x


if __name__ == '__main__':
    tensor = torch.randn((3, 4))  # 生成一个示例张量
    x_min = torch.min(tensor).item()  # 获取最小值的标量值
    x_max = torch.max(tensor).item()  # 获取最大值的标量值

    normalized_tensor = min_max_normalize(tensor, x_min, x_max)
    inverse_tensor = min_max_inverse(normalized_tensor, x_min, x_max)

    print("Original Tensor:")
    print(tensor)

    print("\nNormalized Tensor:")
    print(normalized_tensor)

    print("\nInverse Tensor:")
    print(inverse_tensor)
