# Script used to perform mri normalisation 
###########################################
import nibabel as nib
import sys
import os 

filename = sys.argv[1]
output_dir = sys.argv[2]
orig = nib.load(filename)
I = orig.get_fdata()

subject_filename = os.path.basename(filename).replace("_bias.nii.gz", "_norm.nii.gz")
output_file_path = os.path.join(output_dir, subject_filename)

# scale to 8-bit image pixel range
newmin = 0
newmax = 255  

# linear normalisation function, taken from wikipedia 
Inorm = (I - I.min()) * (newmax - newmin) / (I.max() - I.min()) + newmin

normalised_img = nib.Nifti1Image(Inorm, orig.affine, orig.header)
normalised_img.to_filename(output_file_path)
