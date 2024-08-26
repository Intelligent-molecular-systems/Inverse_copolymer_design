# Graph-to-String Variational Autoencoder for Synthetic Polymer Design - semi-supervised training and property optimized generation of novel polymers
Generative molecular design is becoming an increasingly valuable approach to accelerate materials discovery. Besides comparably small amounts of polymer data, also the complex higher-order structure of synthetic polymers makes generative polymer design highly challenging. We build upon a recent polymer representation that includes stoichiometries and chain architectures of monomer ensembles and develop a novel variational autoencoder (VAE) architecture encoding a graph and decoding a string. Most notably, our model learns a latent space (LS) that enables de-novo generation of copolymer structures including different monomer stoichiometries and chain architectures.

# Steps to train the model and reproduce the results
## Environment setup
Install packages:
- onmt==3.1.3
- torch==1.13.1+cu117
- torch_geometric==2.3.0
- rdkit==2023.03.1
- networkx==2.8.4
- umap-learn==0.5.3

## Create dataloaders
Running the script ```Transform_Batch_Data_g2s.py``` creates the dataloaders that are necessary for training the G2SVAE model. There are multiple options to create different dataloaders depending on tokenization and data augmentation (see comments in script.)
## Training and evaluation
The model can be trained using the script ```train_with_args.py```. See the scripts ```model_runs.sh``` and ```model_run.sbatch``` for automating multiple training runs. 
The scripts ```inference_cluster.py, test_cluster.py, generate.py``` are used to test the model (reconstruction) and generate novel polymers. Afterward, the scripts ```validity.py, check_generated_polymers.py, check_generated_from_seed, plot*.py, ``` can be used to check the reconstructed and novel polymers for the evaluation criteria and for plotting.