# Inverse design of copolymers including chain architecture and stoichiometry


# Steps to train the model and reproduce the results
## Environment setup

## Create dataloaders (folder data_processing)
Running the script ```Transform_Batch_Data_g2s.py``` in the folder dataprocessing/ creates the dataloaders that are necessary for training the G2SVAE model. There are multiple options to create different dataloaders depending on tokenization and data augmentation (see comments in script)
## Training and evaluation
The model can be trained using the script ```train.py```. 
The scripts ```generate.py, test_forward.py, inference_reconstruction.py``` are used to generate novel polymers and test the model (reconstruction). Afterward, the scripts ```reconstruction_validity.py, metrics_generated_polymers.py, plots.py, ``` can be used to check the reconstructed and novel polymers for the evaluation criteria and for plotting.
