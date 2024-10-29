import sys 
import nibabel as nib
import nilearn as nil 
import matplotlib.pyplot as plt
from nilearn import plotting 
import subprocess
import os 

# Locations of DeepBet scripts
skullstrip = "/Users/gabrielhaw/Downloads/DeepBet/muSkullStrip.py"
model = "/Users/gabrielhaw/Downloads/DeepBet/models/Site-All-T-epoch_36.model"

if len(sys.argv) < 3: 
    print("Error: No filename provided.")
    sys.exit(1)

filename = sys.argv[1]
output_dir = sys.argv[2]

output_path = os.path.join(output_dir, "brain_extracted.nii.gz") 
# Using DeepBet-U-Net brain extraction tool
subprocess.run(["python", 
                skullstrip, 
                "-in", filename, 
                "-model", model,
                "-out", output_path
                ])

