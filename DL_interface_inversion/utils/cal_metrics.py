# Calculate root mean square deviation and correlation coefficient for inversion results
import numpy as np
from utils import readGrd
import os


def rmse(a, b):
    # a and b are numpy arrays, can be one or two dimensional
    # Check if shapes of a and b are consistent, if not, raise an exception
    if a.shape != b.shape:
        raise ValueError("The shapes of a and b must be the same.")
    # Calculate the square of the difference between a and b
    diff = (a - b) ** 2
    # Calculate the mean of squared differences to get mean squared error
    mse = np.mean(diff)
    # Take the square root of MSE to get root mean square deviation
    rmsd = np.sqrt(mse)
    # Return root mean square deviation
    return rmsd


def corrcoef(a, b):
    # a and b are one or two dimensional numpy arrays
    # Check if shapes of a and b are consistent, if not, raise an exception
    if a.shape != b.shape:
        raise ValueError("The shapes of a and b must be the same.")
    # Flatten a and b into one-dimensional arrays
    a_flat = a.flatten()
    b_flat = b.flatten()
    # Put a and b into a list
    arrays = [a_flat, b_flat]
    # Calculate correlation coefficient matrix
    corr_matrix = np.corrcoef(arrays)
    # Extract the correlation coefficient between a and b (element at first row, second column)
    corr = corr_matrix[0, 1]
    # Return correlation coefficient
    return corr


if __name__ == '__main__':
    ModNum = 512    # Number of test models
    DensNum = 41    # Number of density steps for each interface
    EvaNum = ModNum * DensNum  # Total evaluation count
    rmse_depth = np.zeros(EvaNum)
    err_density = np.zeros(EvaNum)
    rmse_dg = np.zeros(EvaNum)
    coef = np.zeros(EvaNum)
    testpath = r"D:\Project\DL_interface_inversion\data\Test"

    # Get all predicted model files
    predmod_names = [i for i in os.listdir(os.path.join(testpath, "pred_model")) if i.endswith(".grd")]
    predmod_list = [os.path.join(testpath, "pred_model", i) for i in predmod_names]  # Get all grd files in the pred_model folder
    total_err = []

    for i in range(len(predmod_list)):
        # Inversion result and anomaly file name
        predmod_name = predmod_names[i]
        # Model and density number
        imodel = predmod_name[-23:-18]
        idens = predmod_name[-12:-10]
        true_dens = 0.1 + (int(idens) - 1) * 0.6 / 40.0
        pred_dens = float(predmod_name[-9:-4])
        # print(imodel, idens, true_dens, pred_dens, type(pred_dens))
        # os.system('pause')
        
        # Define file paths
        dg_name = os.path.join(testpath,
                               'dg',
                               'FwGravofBasin_' + imodel + '_Dens_' + idens + '.grd')
        mod_name = os.path.join(testpath,
                                'model',
                                'SedofBasin_' + imodel + '.grd')
        preddg_name = os.path.join(testpath, 'pred_dg', predmod_name)
        
        # Read data
        dg = readGrd.readGrdbynp(dg_name)
        model = readGrd.readGrdbynp(mod_name)
        pred_model = readGrd.readGrdbynp(predmod_list[i])
        pred_dg = readGrd.readGrdbynp(preddg_name)

        # Calculate depth RMSE, density error, gravity anomaly RMSE, and depth correlation coefficient
        rmse_depth[i] = rmse(pred_model, model)
        err_density[i] = abs(pred_dens - true_dens)
        rmse_dg[i] = rmse(pred_dg, dg)
        coef[i] = corrcoef(pred_model, model)
        total_err.append([(i + 1), np.max(model), rmse_depth[i],
                          err_density[i], rmse_dg[i], coef[i],
                          rmse_depth[i] / np.max(model)])

    # Statistical analysis of results
    depth_mean = np.mean(rmse_depth)
    depth_std = np.std(rmse_depth)
    density_mean = np.mean(err_density)
    density_std = np.std(err_density)
    dg_mean = np.mean(rmse_dg)
    dg_std = np.std(rmse_dg)
    coef_mean = np.mean(coef)
    coef_std = np.std(coef)

    # Print statistics
    print(f'depth_mean :{depth_mean}-depth_std===>>{depth_std}')
    print(f'density_mean :{density_mean}-density_std===>>{density_std}')
    print(f'dg_mean :{dg_mean}-dg_std===>>{dg_std}')
    print(f'coef_mean :{coef_mean}-depth_std===>>{coef_std}')
    print(f'coef_min :{np.min(coef)}-depth_max===>>{np.max(coef)}')
    np.savetxt('total_err_1e100_scale_smooth.txt', np.array(total_err))
