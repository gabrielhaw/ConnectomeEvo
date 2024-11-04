# Script used to perform brain extraction of NHPs
#################################################
import sys  
import subprocess

# locations of DeepBet scripts
skullstrip = "/Users/gabrielhaw/Downloads/Working/DeepBet/muSkullStrip.py"
model = "/Users/gabrielhaw/Downloads/Working/DeepBet/models/Site-All-T-epoch_36.model"

if len(sys.argv) < 3: 
    print("Error: No filename provided.")
    sys.exit(1)

filename = sys.argv[1]
output_dir = sys.argv[2]

# using DeepBet-U-Net brain extraction tool
subprocess.run(["python", 
                skullstrip, 
                "-in", filename, 
                "-model", model,
                "-out", output_dir
                ])
