import pandas as pd
import matplotlib.pyplot as plt

# Load data from a space-separated file
errpath = r'D:\Project\DL_interface_inversion\data\Yucca Flat\err_PreDepth64_Dens_scale_1e100_0.587.dat'
# If data is separated differently, modify the sep parameter
df = pd.read_csv(errpath, sep=' ')

# Select specific column (11th column, index 10)
selected_columns = df.iloc[:, 10]

# Flatten the selected column data into a Series
flat_data = selected_columns.values

# Create histogram with scaling factor of 1000
plt.hist(flat_data*1000.0, bins=20, edgecolor='black', density=True)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.xlim(-1000, 1000)  # Set x-axis limits

# Calculate mean and standard deviation
mean_value = flat_data.mean()
std_dev = flat_data.std()

# Add vertical lines for mean and standard deviation boundaries
plt.axvline(mean_value*1000.0, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_value:.2f}')
plt.axvline((mean_value + std_dev)*1000.0, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev = {std_dev:.2f}')
plt.axvline((mean_value - std_dev)*1000.0, color='green', linestyle='dashed', linewidth=2)

# Show legend
plt.legend()

# Add text annotations to the plot
plt.text(-950, 0.007, f'Mean = {mean_value:.2f}', color='red')
plt.text(-950, 0.006, f'Std Dev = {std_dev:.2f}', color='green')

# Display the plot
plt.show()
plt.show()  # Note: Second plt.show() is redundant
