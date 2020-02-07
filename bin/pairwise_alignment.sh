#!/bin/bash

# Path to the script
scd="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
red=$(dirname $scd)


# Add the path to the main folder to the python script
export PYTHONPATH=$PYTHONPATH:$scd

DEFAULT_DATA_DIR="${red}""/data/Audio files (FLAC, 16bit, 44.1k)"
# directory containing recordings
data_dir=${1:-"${DEFAULT_DATA_DIR}"}

# Reference Performance
ref_recording=${2:-"${DEFAULT_DATA_DIR}/Karajan - 1989  (Adagio).flac"}

# Directory for storing alignment results
out_dir=${3:-"${red}/pairwise_alignments"}
# Make directory for results if it does not exist
mkdir -p "${out_dir}"

IFS=$'\n'
for fn in $(ls "${data_dir}"/*.flac); do
    # If the file is not the reference
    if [ $(basename "${fn}") != $(basename "${ref_recording}") ]; then
	# Make alignment
	echo "$(basename ${fn})"
	"${scd}"/align_piece "${fn}" "${ref_recording}" "${out_dir}"
    fi
done
