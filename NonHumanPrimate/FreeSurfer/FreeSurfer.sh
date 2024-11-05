#!/bin/bash
# script implementing steps in freesurfer for NHPs
##########################################################

if [ -z "$1" ] || [ -z "$2" ]; then  
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_dir="$1"
output_dir="$2"

mkdir -p "${output_dir}/freesurfer"
mkdir -p "${input_dir}/formatted"

# so it temporarily goes to directory of your choice
export SUBJECTS_DIR="${output_dir}/freesurfer"

for subject_file in "$input_dir"/*_Christa.aligned.nii.gz; do 
    subject_name=$(basename "$subject_file" .aligned.nii.gz)
    img="$subject_file"

    #mri_convert "$img" "${input_dir}/formatted/${subject_name}.mgz"
    #recon-all -motioncor -s "$subject_name" -i "${input_dir}/formatted/${subject_name}.mgz" -hires
    
    # already performed bias correction but need a nu.mgz file for normalisation
    #cp "${SUBJECTS_DIR}/$subject_name/mri/orig.mgz" "${SUBJECTS_DIR}/$subject_name/mri/nu.mgz"

    #recon-all -s "$subject_name" -autorecon1 -nonuintensitycor -talairach -notal-check -hires -normalization -noskullstrip
    recon-all -s "$subject_name" -autorecon2 -normneck -noskull-lta

done 