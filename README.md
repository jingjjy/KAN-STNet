## KAN-STNet
A Novel High-Dimensional Spatiotemporal Prediction Framework Based on Kolmogorov-Arnold Networks
![Uploading 图片1.png…]()

## Overview
KAN-STNet is a novel high-dimensional spatiotemporal prediction framework that integrates three core technologies: delay embedding theory, spatiotemporal information transformation (STI), and Kolmogorov-Arnold Network (KAN).
It constructs a dual-manifold learning architecture:
Reconstruction manifold
Feature manifold
The model combines:
CNN & MLP for spatial feature extraction
Chebyshev-based KAN for temporal dynamic modeling
STI equations for manifold mapping
KAN-STNet supports accurate multi-step prediction in a single forward pass, enabling efficient and scalable long-term spatiotemporal forecasting.

## Environment Requirements
Python == 3.7.5
h5py == 3.8.0
tensorflow == 2.1.0
numpy == 1.21.6
pillow == 9.5.0
scikit-learn == 1.0.2
pickle == 1.4.1

## Installation & Setup
1. Clone Repository
bash
git clone https://github.com/jingjjy/KAN-STNet.git
cd KAN-STNet
2. Create & Activate Conda Environment
bash
conda create --name KAN_STNet python=3.7.5
conda activate KAN_STNet
3. Install Dependencies
bash
pip install h5py==3.8.0 tensorflow==2.1.0 numpy==1.21.6 pillow==9.5.0 scikit-learn==1.0.2
Note: pickle is included in Python 3.7.5 by default, no need to install separately.
Data Preparation
The dataset is provided in the data/ directory.
Before running, you must modify the data path:
Open data_processing.py
Replace DATA_BASE_DIR with your actual project path

## How to Run
1. Train the Model
bash
python train.py
This will generate model weights.
2. Evaluate the Model
Open eval.py
Replace the path= parameter with your trained weight file path
Run:
bash
python eval.py
Run Time
Installation: A few minutes
Training & Evaluation: Normally completed within a few hours

## License
This project is released under the **MIT License**.
