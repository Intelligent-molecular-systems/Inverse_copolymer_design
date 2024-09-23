# Inverse Design of Copolymers: Stoichiometry and Chain Architecture

**Repository for the paper "Inverse design of copolymers including stoichiometry and chain architecture" [link to arxiv paper]**

This project aims to enable the machine learning-guided discovery of novel copolymers by generating monomer ensembles with different stoichiometries and chain architectures. Our model is based on a novel Variational Autoencoder (VAE) that encodes a graph and decodes a string, offering advanced polymer design capabilities, including inverse design and optimization in latent space.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Creating Dataloaders](#creating-dataloaders)
  - [Training the Model](#training-the-model)
  - [Evaluation](#evaluation)
  - [Optimization in Latent Space](#optimization-in-latent-space)
  - [Plotting Results](#plotting-results)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
Synthetic polymer discovery is a challenging task due to the vast design space and the hierarchical complexity of polymer structures. This project advances machine learning methods by using a semi-supervised VAE for copolymer generation. Key contributions include:
- Generation of monomer ensembles with stoichiometry and chain architecture.
- A novel VAE architecture capable of encoding a graph and decoding a string.
- Optimization of copolymers in latent space for specific electronic properties like electron affinity and ionization potential, demonstrated for photocatalysts in hydrogen production.

## Installation
To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/GaVogel/G2SVAE
cd G2SVAE
conda env create -f environment.yml
conda activate G2S_VAE_env
```
Alternatively, install the packages manually: 
- onmt==3.1.3
- torch==1.13.1+cu117
- torch_geometric==2.3.0
- rdkit==2023.03.1
- networkx==2.8.4
- umap-learn==0.5.3
- pandas==2.0.1
- matplotlib==3.7.1
- bayesian-optimization=1.4.3
- pymoo=0.6.1.1

## System Requirements
The code was tested on windows and linux with Python 3.10.11 and a cuda-enabled GPU.

## Project structure
The repository is structured as follows
```
.
├── Checkpoints/           # Saved models and results
├── data/                  # Dataset for training and testing, dataloaders
├── data_processing/       # Data handling, creating dataloaders 
├── model/                 # Graph-to-String VAE architecture
├── scripts/               # Scripts for training, testing, and plotting 
└── README.md              # Project documentation 
```

## Usage

### 1. Creating Dataloaders
Running the script ```Transform_Batch_Data_g2s.py``` creates the dataloaders that are necessary for training the G2SVAE model. There are multiple options to create different dataloaders depending on tokenization and data augmentation (see comments in script.)

### 2. Training the Model
Train the VAE using the provided training script. The model learns a continuous latent space for polymer representation. See the script for arguments regarding the training (augmentation, tokenization, loss hyperparameters, etc.). 

```bash
python scripts/train.py
```
This will save the trained model checkpoints in the `Checkpoints/<model name>` folder.

### 3. Evaluation
To test the model's performance, we first can run it on unseen data 

```bash
python scripts/test_forward.py
python scripts/inference_reconstruction.py
```
Further, the script ``` generate.py``` is used to generate novel polymers. Afterward, the scripts ```reconstruction_validity.py, metrics_generated_polymers.py, plots.py, ``` can be used to check the reconstructed and novel polymers for the evaluation criteria and for plotting. </br>
The arguments must match the arguments used for training the model. 

### 4. Optimization in Latent Space
To perform inverse design via optimization in latent space, you can use two optimization scripts (Bayesian optimization or Genetic Algortihm).

```bash
python scripts/optimize_BO.py
python scripts/optimize_GA_correct.py
```

The scripts will explore the latent space to find new copolymers with the desired properties. For the property target values check the scripts.

### 5. Plotting Results
The directory "plotting/" contains further scripts for analysis and visualization. 

## Results


## Contributing
We welcome contributions to improve the codebase or extend the functionality of the model.
1. Fork the repository
2. Create a new branch 
3. Make your changes
4. Commit your changes 
5. Push to the branch 
6. Open a pull request

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.


