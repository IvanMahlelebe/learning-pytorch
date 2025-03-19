# Learning PyTorch

## Repository Purpose

This repository serves as a version-controlled record of my journey in learning PyTorch. I am using the official [PyTorch tutorials](https://pytorch.org/tutorials/index.html) to supplement my theoretical knowledge of deep learning with practical experience using frameworks.

## Setup Instructions

To run the code in this repository, follow these steps:

### 1. Create the Conda Environment

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed. Then, create the environment using the provided `environment.yml` file:

```sh
conda env create -f environment.yml
```

Activate the newly created environment:

```sh
conda activate pytorch
```

### 2. Install PyTorch

Since PyTorch is not included in the `environment.yml` file, install it manually using the following command:

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

This command installs PyTorch and its dependencies for CPU usage. If you have a GPU, refer to the [official installation guide](https://pytorch.org/get-started/locally/) to install the appropriate version.

## Repository Link

Find the latest updates and code in this repository: [learning-pytorch](https://github.com/IvanMahlelebe/learning-pytorch.git)


---
**Author:** IvanMahlelebe
