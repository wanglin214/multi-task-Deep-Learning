import numpy as np
import os


def outGrd(filepath, grdata, gdmin, gdmax):
    """
    Save data to a Surfer 6 Text format GRD file
    
    Parameters:
    -----------
    filepath : str
        Path where the GRD file will be saved
    grdata : numpy.ndarray
        2D array containing the grid data to be saved
    gdmin : float
        Minimum value in the grid data
    gdmax : float
        Maximum value in the grid data
    
    Returns:
    --------
    None
    """
    # print(grdata.shape, type(grdata), Nx,Ny)
    # print(filepath)
    # os.system('pause')
    
    # Alternative method for writing GRD files (commented out)
    # with open(filepath, 'w') as f:
    #     f.write('DSAA\n')
    #     f.write('{0} {1}\n'.format(Nx, Ny))
    #     f.write('{0} {1}\n'.format(Xmin, Xmax))
    #     f.write('{0} {1}\n'.format(Ymin, Ymax))
    #     f.write('{0} {1}\n'.format(gdmin, gdmax))
    #
    # f.close()
    
    # Create the GRD header with fixed grid size (64x64) and domain (0.0-40.0 km)
    header = "DSAA\n 64 64\n 0.0 40.0\n 0.0 40.0\n " + str(gdmin) + ' ' + str(gdmax)
    # print(header)
    # os.system('pause')

    # Save the data with the header, no comment character, and fixed format
    np.savetxt(filepath, grdata, header=header, comments='', fmt='%8.3f')

    return
