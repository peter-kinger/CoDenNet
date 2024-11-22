# CoDenNet
this is a repository about the code of CoDenNet
Coupled Dense Convolutional Neural Networks with Autoencoder for Unsupervised Hyperspectral Super-Resolution
This repository contains the implementation of the Coupled Dense Convolutional Neural Networks with Autoencoder model for unsupervised hyperspectral super-resolution, as described in the associated paper. The goal of this work is to enhance the spatial resolution of hyperspectral images (HSI) by leveraging dense CNNs and autoencoders in an unsupervised learning framework.

Table of Contents
Introduction
Requirements
Datasets
Usage
Training
Testing
Files Overview
Citation
License
Introduction
Hyperspectral imaging (HSI) provides rich spectral information, but often suffers from low spatial resolution. This work proposes a novel framework combining dense convolutional neural networks and autoencoders to super-resolve hyperspectral images without requiring labeled data. Key features include:

Coupled Dense Networks: Improve spatial resolution by extracting multi-scale features.
Autoencoder: Ensures spectral consistency and dimensionality reduction.
Unsupervised Learning: Eliminates dependency on large labeled datasets.
Requirements
To set up the environment, install the dependencies listed below:

Python 3.8+
PyTorch 1.9+
NumPy
Matplotlib
OpenCV (cv2)
Install dependencies using:

bash
复制代码
pip install -r requirements.txt
Datasets
This repository supports multiple datasets, including:

Indian Pines (9003): A widely-used hyperspectral dataset for testing and benchmarking.
The dataset preprocessing steps are described in the files HSI_origin.md and MSI_origin.md. Ensure datasets are placed in the correct directories before training or testing.

Usage
Training
To train the model, run the following command:

bash
复制代码
python train.py --dataset_path <path_to_dataset> --output_dir <output_directory> --epochs <num_epochs>
Testing
To evaluate the model on a test set, use:

bash
复制代码
python test.py --model_path <path_to_model> --test_data <path_to_test_data>
Alternatively, the newtest.py script can be used for extended testing functionalities.

Example: Indian Pines
The script bash_indian_9003.sh automates training and testing for the Indian Pines dataset. Execute it using:

bash
复制代码
bash bash_indian_9003.sh
Files Overview
9003.txt: A sample or configuration file specific to the Indian Pines dataset.
HSI_origin.md / MSI_origin.md: Documentation for dataset preparation.
train.py: Script for training the model.
test.py / newtest.py: Scripts for testing and evaluation.
bash_indian_9003.sh: Bash script for running experiments on the Indian Pines dataset.
