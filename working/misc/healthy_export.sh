#!/bin/bash

# Command line arguments
bio_file=$1
subject_directory=$2
output_directory=$3
id1=$4
id2=$5

# Help message: check for required arguments
if [ -z "$bio_file" ] || [ -z "$subject_directory" ] || [ -z "$output_directory" ] || [ -z "$id1" ] || [ -z "$id2" ]; then
    echo "Usage: $0 bio_file subject_directory output_directory id1 id2"
    exit 1
fi

# Check (and create) the output directory if it doesn't exist
if [ ! -d "$output_directory" ]; then
    echo "Output directory does not exist, creating $output_directory"
    mkdir -p "$output_directory"
fi

# Define a file to store the result CSV
result_csv="$output_directory/results.csv"

# Use awk to search the CSV file by header names.
# The CSV file should have a header with at least these columns:
# participant_id, study, dx, etiv, age, sex
#
# The script does the following:
# 1. In the header row (NR==1), it finds the column indices for all required headers.
# 2. It prints a header to the result CSV.
# 3. For each subsequent row, if study == id1 and dx == id2,
#    it prints the row's participant_id, etiv, age, and sex values to the result CSV.
awk -F, -v id1="$id1" -v id2="$id2" '
NR==1 {
    # Loop through header fields to find column positions
    for (i = 1; i <= NF; i++) {
        # Remove potential quotes and extra spaces
        gsub(/^ *"|" *$/, "", $i)
        if ($i == "participant_id") { pid_idx = i }
        else if ($i == "study") { study_idx = i }
        else if ($i == "dx") { dx_idx = i }
        else if ($i == "etiv") { etiv_idx = i }
        else if ($i == "age") { age_idx = i }
        else if ($i == "sex") { sex_idx = i }
    }
    # Ensure required columns exist
    if (!pid_idx || !study_idx || !dx_idx || !etiv_idx || !age_idx || !sex_idx) {
        print "ERROR: Required columns not found" > "/dev/stderr"
        exit 1
    }
    # Print header for result CSV
    print "participant_id,etiv,age,sex"
    next
}
{
    # Trim extra spaces from study and dx fields
    gsub(/^ *| *$/, "", $study_idx)
    gsub(/^ *| *$/, "", $dx_idx)
    if ($study_idx == id1 && $dx_idx == id2) {
        # Print matching row's participant_id, etiv, age, and sex
        print $(pid_idx) "," $(etiv_idx) "," $(age_idx) "," $(sex_idx)
    }
}
' "$bio_file" > "$result_csv"

echo "Result CSV created at: $result_csv"

# Extract participant_ids from the result CSV (skip the header)
participant_ids=()
while IFS=',' read -r participant_id _; do
    # Skip header row if it appears
    if [ "$participant_id" != "participant_id" ]; then
        participant_ids+=("$participant_id")
    fi
done < <(tail -n +2 "$result_csv")

# Debug: List found participant IDs
echo "Found participant IDs:"
for pid in "${participant_ids[@]}"; do
    echo "$pid"
done

# Move folders corresponding to the participant IDs from subject_directory to output_directory
for participant_id in "${participant_ids[@]}"; do
    src_folder="$subject_directory/$participant_id"
    if [ -d "$src_folder" ]; then
        mv "$src_folder" "$output_directory"
        echo "Moved folder for participant $participant_id to $output_directory"
    else
        echo "Folder for participant $participant_id does not exist in $subject_directory"
    fi
done
