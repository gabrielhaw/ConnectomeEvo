#!/bin/bash
# bash script that runs the NHP preprocessing scripts with optional cleanup
###########################################################################

if [ -z "$1" ] || [ -z "$2" ]; then 
    echo "Usage: $0 <input_directory> <output_directory> [-c]"
    exit 1
fi 

input_dir="$1"
output_dir="$2"
template_file="/Users/gabrielhaw/Downloads/Working/Juna.Chimp_05mm/Juna_Chimp_T1_05mm_skull_stripped.nii"
clean_up=0

if [[ "$3" == "-c" ]]; then
    clean_up=1
    echo "clean up of intermediate files and directories selected"
else
    echo "no clean up of intermediate files"
fi

echo "template file: $template_file"

mkdir -p "${output_dir}/aligned"
mkdir -p "${output_dir}/aligned/matrices"
mkdir -p "${output_dir}/intermediate/brain_extracted"
mkdir -p "${output_dir}/intermediate/pre_mask"
mkdir -p "${output_dir}/intermediate/corrected"

for subject_file in "${input_dir}"/*.nii; do 
    subject_name=$(basename "$subject_file" .nii)
    img="$subject_file"
    echo "processing subject: $subject_name"

    # image correction steps
    corrected_img="${output_dir}/intermediate/corrected/${subject_name}.nii.gz"
    DenoiseImage -d 3 -i "$img" -o "$corrected_img"
    N4BiasFieldCorrection -d 3 -i "$corrected_img" -o "$corrected_img"
    python3 normalise.py "$corrected_img" "${output_dir}/intermediate/corrected/"
    SmoothImage 3 "$corrected_img" 0.4 "$corrected_img"    
    
    # generate brain mask
    python3 background_removal.py "$corrected_img" "${output_dir}/intermediate/pre_mask"

    # extract the brain 
    brain_extracted="${output_dir}/intermediate/brain_extracted/${subject_name}.nii.gz"
    fslmaths "$corrected_img" -mas "${output_dir}/intermediate/pre_mask/${subject_name}_pre_mask.nii.gz" "$brain_extracted"

    # affine alignment to template orientation
    affine_output="${output_dir}/aligned/${subject_name}.nii.gz"
    affine_matrix="${output_dir}/aligned/matrices/${subject_name}_affine_matrix.mat"
    affine_log="${output_dir}/aligned/matrices/${subject_name}_affine_to_Juna_Chimp_T1_05mm_skull_stripped.log"
    flirt -in "$brain_extracted" -ref "$template_file" -out "$affine_output" -omat "$affine_matrix" -cost mutualinfo -dof 7

done

# optional cleanup of directories 
if [ "$clean_up" -eq 1 ]; then
    echo "cleaning up folders..."
    rm -rf "${output_dir}/intermediate/" 
fi

echo "Processing completed for all subjects."
