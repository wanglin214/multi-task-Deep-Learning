# Calculate the root mean square deviation and interface correlation coefficient for inversion results
import numpy as np
import os

if __name__ == '__main__':
    testpath = r"D:\Project\DL_interface_inversion\data\Test"
    
    # Get all predicted model files
    predmod_names = [i for i in os.listdir(os.path.join(testpath, "pred_model")) if i.endswith(".grd")]
    predmod_list = [os.path.join(testpath, "pred_model", i) for i in predmod_names]  # Get all grd files in the pred_model folder
    
    # Extract density values from filenames
    dens = np.zeros(len(predmod_list))
    for i in range(len(predmod_list)):
        dens[i] = float(predmod_names[i][-9:-4])
        # print(dens[i])
        # os.system('pause')

    print(len(predmod_list), predmod_list[0])
    
    # Write model names and density values to a text file
    with open("dir_grd_40km_scale_smooth.txt", "w") as file:
        # Write the string list to the file, one string per line
        for i in range(len(predmod_list)):
            file.write(predmod_names[i] + ' ' + str(dens[i]) + "\n")

    print("String list has been saved to dir_grd_40km_scale_smooth.txt file.")
