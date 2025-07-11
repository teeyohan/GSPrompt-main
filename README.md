# GSPrompt-main

PyTorch implementation of "Learning Generic and Specific Prompts with Contrastive Constraints for Multi-task Visual Scene Understanding".

<div align="center">
  <img width="80%" alt="" src="GSPrompt.png">
</div>

## Setup
The following dependencies are recommended for the installation of the environment.

- python 3.7.16
- torch 1.10.0
- torchvision 0.11.0

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

The optimal-dataset-scale F-measure (odsF) of boundary detection is evaluated using this tools:
- Evaluation Tools: https://github.com/prismformore/Boundary-Detection-Evaluation-Tools
