# Frame Selection for vLLMs via Submodular Optimization

## Overview

This repository implements frame selection for video language models (vLLMs) using submodular optimization techniques. The evaluation framework is built on top of the [lmms-eval repository](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Setup

The evaluation code uses lmms-eval as the underlying framework. For installation instructions, please refer to the [lmms-eval documentation](https://github.com/EvolvingLMMs-Lab/lmms-eval).

**Note:** lmms-eval serves as a black-box interface for evaluating different vLLMs across various tasks.

## Frame Selection Pipeline

The `frame-selection` folder contains three core Python scripts that implement the frame extraction and selection pipeline:

### 1. `feature_extract.py`
Extracts visual features and computes text-frame similarities from video data.

**Functionality:**
- Takes video paths and text data as input
- Uses CLIP model (or alternative models) to extract frame-level features
- Computes text-frame similarity scores
- Saves feature embeddings and similarity scores to the `output_features` folder

**Input format for LongVideoBench dataset:**
- `*_lvb_val.json`: Contains text queries and video file paths
- `*_videos/`: Directory containing the video files

### 2. `semantic_tags_extract.py`
Computes similarity scores between frames and semantic tags derived from text queries.

**Functionality:**
- Takes video paths and text data as input
- Uses pretrained models to extract semantic tags from text query
- Employs CLIP model (or alternative models) to calculate frame-tag similarity scores
- Stores results in the `output_features` folder
- Output is a pickle file of list with each element being a dictionary of {'importance_scores':,
                'similarity_matrix':}, corresponding to a single video and query task. Here 'importance_scores' is a list of keywords score where similarity matrix is a numpy array of dimensions frame_nums \times num_of_tags 

### 3. `frame_select.py`
Performs intelligent frame selection using submodular optimization.

**Functionality:**
- Loads features and similarity scores from previous steps
- Calculates or loads pairwise similarity between frames
- Selects optimal frames via submodular maximization under cardinality constraints

## Workflow

1. Extract frame features and text-frame similarities (`feature_extract.py`)
2. Compute semantic tag similarities (`semantic_tags_extract.py`)
3. Select optimal frames using submodular optimization (`frame_select.py`)

## Citation

If you use this code in your research, please cite the original lmms-eval repository and this work.