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

for subject_file in "$input_dir"/*.nii.gz; do 
    subject_name=$(basename "$subject_file" .nii.gz)
    img="$subject_file"

    # expected orientation 
    orient="${input_dir}/formatted/${subject_name}.mgz"
    mri_convert "$img" "$orient"

    # nuintensity correction makes images too bright
    mri="${output_dir}/freesurfer/${subject_name}/mri"
    cp "${mri}/orig.mgz" "${mri}/nu.mgz"
    
    # to start freesurfer process
    recon-all -motioncor  -s "$subject_name" -i "${input_dir}/formatted/${subject_name}.mgz" -hires

    # need to check if i can move this up and replace it with mri_convert
    mri_convert --in_orientation LIA --out_orientation RAS "${mri}/orig.mgz" "${mri}/orig.mgz"

    # already performed skull stripping
    cp "${mri}/T1.mgz" "${mri}/brainmask.mgz"

    recon-all -s "$subject_name" -autorecon1 -notal-check -hires -noskullstrip -nonuintensitycor
    recon-all -s "$subject_name" -autorecon2 -normneck -noskull-lta -hires

done 