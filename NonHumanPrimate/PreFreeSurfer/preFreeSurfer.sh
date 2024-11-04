#!/bin/bash
# Bash script that runs the NHP preprocessing scripts
#####################################################

if [ -z "$1" ] || [ -z "$2" ]; then 
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi 

input_dir="$1"
output_dir="$2"
template_file="/Users/gabrielhaw/Downloads/Working/Juna.Chimp_05mm/Juna_Chimp_T1_05mm_skull_stripped.nii"

echo "Template File: $template_file"


mkdir -p "${output_dir}/pre_masks"
mkdir -p "${output_dir}/brain_extracted"

for subject_file in "${input_dir}"/test.nii; do 
    subject_name=$(basename "$subject_file" .nii)
    img="$subject_file"
    echo "Processing Subject: $subject_name"

    # image correction steps
    DenoiseImage -d 3 -i "$img" -o "$img"
    N4BiasFieldCorrection -d 3 -i "$img" -o "$img"

    # generate brain mask
    python3 background_removal.py "$img" "${output_dir}"

    # extract the brain 
    fslmaths "$img" -mas "${output_dir}/${subject_name}_pre_mask.nii.gz/${subject_name}_pre_mask.nii.gz" "${output_dir}/brain_extracted/${subject_name}.nii.gz"

    # rough alignment to template orientation
    flirt -in "${output_dir}/brain_extracted/${subject_name}.nii.gz" -ref "$template_file" -out "${output_dir}/${subject_name}.aligned.nii.gz" -cost mutualinfo -dof 6

done

echo "Processing completed for all subjects."
