# Batch rename files, just need to set the renamed file names in the script
import os

# Usage example: Add prefix "new_" and suffix "_data" to all .grd files in the directory and move them to a new directory
inpath = r"E:\frequency_interface_forward\FmtGrd"
outpath = r"E:\frequency_interface_forward\Basement"

# Get all files in the input directory
files = os.listdir(inpath)
# print(files.__len__(), type(files))
Nfiles = files.__len__()

i = 0

for filename in files:
    # Build the complete input file path
    old_path = os.path.join(inpath, filename)
    i = i+1
    
    # Check if the file is a file and not a directory
    if os.path.isfile(old_path):
        # Build the new filename
        # Format i as a 5-digit number with leading zeros
        new_filename = 'SedofBasin_' + "{:05d}".format(i) + '.grd'
        
        # Build the complete output file path
        new_path = os.path.join(outpath, new_filename)
        
        # Perform the rename and move operation
        os.rename(old_path, new_path)
        print(f'Renamed and moved: {filename} -> {new_filename} to {outpath}')
