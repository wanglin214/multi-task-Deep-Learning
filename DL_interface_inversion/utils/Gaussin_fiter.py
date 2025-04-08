import os
import numpy as np
from scipy.ndimage import gaussian_filter
from utils import readGrd, outGrd

# Import matplotlib library
import matplotlib.pyplot as plt

sigma = 4  # Gaussian filter standard deviation
myroot = r"D:\Project\DL_interface_inversion\data\field\bishop64x_basement_km.grd"  # Input file path
filepath = r"D:\Project\DL_interface_inversion\data\field\bishop64x_basement_km_filter.grd"  # Output file path

if __name__ == '__main__':  

    dg = readGrd.readGrdbynp(myroot)  # Read grid data as numpy array

    # Apply Gaussian filter to the data
    filtered_data = gaussian_filter(dg, sigma)

    # Define contour levels and colormap
    levels = 10
    cmap = plt.cm.coolwarm

    # Create canvas and subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot filled contours of original data
    ax[0].contourf(dg, levels, cmap=cmap)
    ax[0].set_title("Original data")

    # Plot filled contours of filtered data
    cs = ax[1].contourf(filtered_data, levels, cmap=cmap)
    ax[1].set_title("Filtered data")

    # Add a colorbar to the figure using the contour set from the filtered data
    fig.colorbar(cs, shrink=0.8)

    # Display the figure
    plt.show()

    # Save the filtered data to a grid file
    outGrd.outGrd(filepath, filtered_data, np.min(filtered_data), np.max(filtered_data))
