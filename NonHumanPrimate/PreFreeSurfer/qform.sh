#!/bin/bash
# script to make sure the qform is set
#####################################
IMAGE_DIR="$1"

for img in "$IMAGE_DIR"/*.nii*; do
    img_name=$(basename "$img")

    qform_code=$(fslhd "$img" | grep '^qform_code' | awk '{print $2}')

    if [ "$qform_code" -eq 0 ]; then
        fslorient -setqformcode 1 "$img"

    fi
done

