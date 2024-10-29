#!/bin/bash

if [-z "$1"]; then 
    echo "Usage: $0 <filename>"
    exit 1

fi 

filename = $1
outputdir = $2
python3 background_removal.py "$filename" "$outputdir"
