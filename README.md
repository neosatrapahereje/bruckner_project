# Bruckner Project

This repository contains tools for audio-to-audio alignments.

## Set up

1. Set up the Python environment (using `conda`).
```bash
conda env create -f environment.yml
```

2. Compile Cython code

```bash
# Activate python environment
conda activate bruckner
# Compile cython code
python setup.py build_ext --inplace
```

2. Get dataset

## Running alignments

To align a single performance

```bash
#Activate python environment
conda activate bruckner
#run code
./bin/align_piece path/to/performance_to_align.flac path/to/reference_performance.flac output_dir
```

For more information
```
./bin/align_piece -h
```
