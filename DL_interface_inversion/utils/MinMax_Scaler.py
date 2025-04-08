import torch


# Define min-max normalization function
def min_max_normalize(x, x_min, x_max):
    # Apply normalization formula
    x_norm = (x - x_min) / (x_max - x_min)

    # Return normalized data
    return x_norm


# Define inverse normalization function
def min_max_inverse(x_norm, x_min, x_max):
    # Apply inverse normalization formula
    x = x_norm * (x_max - x_min) + x_min

    # Return denormalized data
    return x


if __name__ == '__main__':
    tensor = torch.randn((3, 4))  # Generate an example tensor
    x_min = torch.min(tensor).item()  # Get the minimum value as a scalar
    x_max = torch.max(tensor).item()  # Get the maximum value as a scalar

    normalized_tensor = min_max_normalize(tensor, x_min, x_max)
    inverse_tensor = min_max_inverse(normalized_tensor, x_min, x_max)

    print("Original Tensor:")
    print(tensor)

    print("\nNormalized Tensor:")
    print(normalized_tensor)

    print("\nInverse Tensor:")
    print(inverse_tensor)
