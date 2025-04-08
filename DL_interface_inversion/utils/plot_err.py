import pandas as pd
import matplotlib.pyplot as plt

# Read data
file_path = r'D:\Project\DL_interface_inversion\test_log_depth_e200_40km.txt'  # Replace with your file path
data = pd.read_csv(file_path, sep='\s+', header=None)  # Read using whitespace separator, no header

# Separate data
x = data.iloc[:, 0]  # First column as x-axis data
y = data.iloc[:, 3]  # Fourth column as y-axis data

# Create bar chart
plt.bar(x, y, color='blue', alpha=0.7)  # Blue bars with 0.7 transparency

# Add axis labels
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Add title
plt.title('Bar Chart of Column 1 and Column 3')

# Display the chart
plt.show()
