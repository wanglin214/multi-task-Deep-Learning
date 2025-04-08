## Read gravity anomaly GRD file and residual density model file
# @Time: 2023/2/17 10:59
# @Author: WangLin
# @File: ReadGrd.py: Read GRD file
# @Software: PyCharm

import os
import numpy as np


# Function to read GRD file, returns grid data array grdata(Ny,Nx), where Ny is the number of columns - corresponding to dim=0
def readGrdbynp(filepath):
    with open(filepath, "r", encoding="UTF-8") as infile:
        infile.readline()  # Skip the first line of standard Surfer 6 text format file with DSAA annotation
        str = infile.readline().split()  # Read the second line with number of points and lines
        Nx = int(str[0])
        Ny = int(str[1])
        # print(Nx, Ny)
        str = infile.readline().split()  # Read the third line with X direction min and max values
        Xmin = float(str[0])
        Xmax = float(str[1])
        # print(Xmin, Xmax)
        str = infile.readline().split()  # Read the fourth line with Y direction min and max values
        Ymin = float(str[0])
        Ymax = float(str[1])
        str = infile.readline().split()  # Read the fifth line with min and max values of grid data
        gdmin = float(str[0])
        gdmax = float(str[1])
        # print(gdmin, gdmax)
    infile.close()
    gd = np.loadtxt(filepath, skiprows=5)  # Load the actual grid data, skipping the header (5 lines)
    # print(gd.shape, gd.min(), gd.max())
    # os.system('pause')
    return gd


# Local subroutine test
if __name__ == '__main__':
    Fgafile = r"D:\Project\DL_interface_inversion\origin_data\dg\FwGravofBasin_0001_Dens_01.grd"
    grddata = readGrdbynp(Fgafile)
    print(grddata.shape)
    print(grddata.min())
