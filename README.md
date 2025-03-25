# GSPrompt-main

PyTorch implementation of "Integrating Generic and Specific Prompts for Multi-task Visual Scene Understanding".

<div align="center">
  <img width="80%" alt="" src="GSPrompt.png">
</div>

## Setup
The dependencies are in requirements.txt. Python=3.8 is recommended for the installation of the environment.

## Dataset
It is recommended to download the following datasets from the official website:

- MNIST: https://yann.lecun.com.
- IMDB: https://developer.imdb.com.
- BACE: https://moleculenet.org.

, and place them in the "BETA-main/data/" directory.

## Training and evaluation
For training and evaluation, use the following script:

- `python main.py`

, where the perturbation magnitude can be estimated using:

- `python determination.py`
