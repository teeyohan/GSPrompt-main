# GSPrompt-main

PyTorch implementation of "Integrating Generic and Specific Prompts for Multi-task Visual Scene Understanding".

<div align="center">
  <img width="80%" alt="" src="GSPrompt.png">
</div>

## Setup
Python=3.7 with torch=1.10.0+cu111 and torchvision=0.11.0+cu111 are recommended for the installation of the environment.

## Dataset
It is recommended to download the following datasets from the official website:

- NYUDv2: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- PASCAL-Context: https://cs.stanford.edu/~roozbeh/pascal-context/

, and place them in the "GSPrompt-main/data/" directory.

## Training and evaluation
For training and evaluation, use the following script:

For NYUDv2:
- `bash run_gsp_nyud.sh`

For PASCAL-Context:
- `bash run_gsp_pascal.sh`

Where the optimal dataset F-measure scores of boundary detection is evaluated used tools in 

- Evaluation Tools: https://github.com/prismformore/Boundary-Detection-Evaluation-Tools
