# Inverse design of copolymers including chain architecture and stoichiometry


# Steps to train the model and reproduce the results
## Environment setup

## Create dataloaders (folder data_processing)
Running the script ```Transform_Batch_Data_g2s.py``` in the folder dataprocessing/ creates the dataloaders that are necessary for training the G2SVAE model. There are multiple options to create different dataloaders depending on tokenization and data augmentation (see comments in script)
## Training and evaluation
The model can be trained using the script ```train_with_args.py```. See the scripts ```model_runs.sh``` and ```model_run.sbatch``` for automating multiple training runs. 
The scripts ```inference_cluster.py, test_cluster.py, generate.py``` are used to test the model (reconstruction) and generate novel polymers. Afterward, the scripts ```validity.py, check_generated_polymers.py, check_generated_from_seed, plot*.py, ``` can be used to check the reconstructed and novel polymers for the evaluation criteria and for plotting.