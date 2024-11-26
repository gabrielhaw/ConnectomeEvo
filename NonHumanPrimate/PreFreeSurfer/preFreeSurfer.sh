#!/bin/bash
# bash script that runs the NHP preprocessing scripts with optional cleanup
###########################################################################

if [ -z "$1" ] || [ -z "$2" ]; then 
    echo "Usage: $0 <input_directory> <output_directory> [-c]"
    exit 1
fi 

input_dir="$1"
output_dir="$2"
template_brain="/Users/gabrielhaw/Downloads/Working/Juna.Chimp_05mm/Juna_Chimp_T1_05mm_skull_stripped.nii.gz"
clean_up=0

if [[ "$3" == "-c" ]]; then
    clean_up=1
    echo "clean up of intermediate files and directories selected"
else
    echo "no clean up of intermediate files"
fi

echo "template file: $template_brain"

mkdir -p "${output_dir}/final"
mkdir -p "${output_dir}/intermediate"

for subject_file in "${input_dir}"/*.nii; do 
    subject_name=$(basename "$subject_file" .nii)
    img="$subject_file"
    echo "Processing subject: $subject_name"

    corrected_img="${output_dir}/intermediate/${subject_name}.nii.gz"
    resampled_img="${output_dir}/intermediate/${subject_name}.nii.gz"

    # denoising, bias correction, and resampling
    DenoiseImage -d 3 -i "$img" -o "$corrected_img"
    N4BiasFieldCorrection -d 3 -i "$corrected_img" -o "$corrected_img"
    ResampleImage 3 "$corrected_img" "$resampled_img" 0.5x0.5x0.5 0

    # smoothing and normalization
    SmoothImage 3 "$resampled_img" 0.4 "$resampled_img"
    python3 normalise.py "$resampled_img" "${output_dir}/intermediate/"

    # generate brain mask and extract brain
    python3 background_removal.py "$resampled_img" "${output_dir}/intermediate/"
    mask="${output_dir}/intermediate/${subject_name}_pre_mask.nii.gz"
    brain_extracted="${output_dir}/intermediate/${subject_name}_extracted.nii.gz"
    fslmaths "$resampled_img" -mas "$mask" "$brain_extracted"

    # final alignment to brain template
    reg_output="${output_dir}/final/${subject_name}_aligned.nii.gz"
    reg_matrix="${output_dir}/intermediate/${subject_name}_reg_matrix.mat"
    flirt -in "$brain_extracted" -ref "$template_brain" -out "$reg_output" -omat "$reg_matrix" -cost corratio -dof 6

    # cleanup intermediate files
    if [ "$clean_up" -eq 1 ]; then
        echo "Cleaning up files..."
        rm -rf "$corrected_img" "$resampled_img" "$brain_extracted" "$mask"
    fi

done

# optional cleanup of directories 
if [ "$clean_up" -eq 1 ]; then
    echo "cleaning up folders..."
    rm -rf "${output_dir}/intermediate"  "${output_dir}/corrected"
fi

echo "Processing completed for all subjects."
