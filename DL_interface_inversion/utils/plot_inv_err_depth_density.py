import pandas as pd
import matplotlib.pyplot as plt

# Load data from a space-separated file
errpath = r'D:\Project\DL_interface_inversion\utils\total_err_1e100_scale.txt'
# If data is separated differently, modify the sep parameter
df = pd.read_csv(errpath, sep=' ')

# Select specific column (7th column, index 6)
selected_columns = df.iloc[:, 6]

# Flatten the selected column data into a Series
flat_data = selected_columns.values

# Create histogram
plt.hist(flat_data, bins=20, edgecolor='black', density=True)

# Set x-axis label
# Multiple label options are commented out - uncomment the one needed
# plt.xlabel('RMSE of depth inversion/m')
# plt.xlabel('ABSE of density inversion/(g/cm³)')
# plt.xlabel('RMSE of reconstructed gravity anomaly/mGal')
# plt.xlabel('Morphology Similarity')
plt.xlabel('Relative RMSE of depth inversion')

# Set y-axis label
plt.ylabel('Frequency')

# Title is commented out
# plt.title('Histogram')

# X-axis limits are commented out
# plt.xlim(-1000, 1000)

# Display the plot
plt.show()
 
