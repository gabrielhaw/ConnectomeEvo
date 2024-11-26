#!/bin/bash
# script implementing steps in freesurfer for NHPs (parallel)
##########################################################

if [ -z "$1" ] || [ -z "$2" ]; then  
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_dir="$1"
output_dir="$2"

mkdir -p "${output_dir}/freesurfer"
mkdir -p "${input_dir}/formatted"
export SUBJECTS_DIR="${output_dir}/freesurfer"

# format input files to .mgz and ensure proper orientation
for subject_file in "$input_dir"/*.nii.gz; do 
    subject_name=$(basename "$subject_file" .nii.gz)
    formatted_img="${input_dir}/formatted/${subject_name}.mgz"
    mri_convert --in_orientation LIA  --out_orientation RAS "$subject_file" "$formatted_img"
done

# nu correction didnt perform well, so we just want orig.mgz 
ls "${input_dir}/formatted/"*.mgz | parallel --jobs 6 \
 recon-all -motioncor -hires -i {} -s {/.} 

# replace nu.mgz with orig.mgz for each subject to skip nu correction
for subject_dir in "${SUBJECTS_DIR}"/*; do
    mri="${subject_dir}/mri"
    cp "${mri}/orig.mgz" "${mri}/nu.mgz"
done

# skip skull stripping and intensity correction
ls "${SUBJECTS_DIR}" | parallel --jobs 6 \
 recon-all -s {} -autorecon1 -hires -notal-check -noskullstrip -nonuintensitycor

# already performed skull stripping
for subject_dir in "${SUBJECTS_DIR}"/*; do
    mri="${subject_dir}/mri"
    cp "${mri}/T1.mgz" "${mri}/brainmask.mgz"
done

# already extracted brain
ls "${SUBJECTS_DIR}" | parallel --jobs 6 \
 recon-all -s {} -autorecon2 -normneck -noskull-lta -hires

echo "Processing complete for all subjects!"