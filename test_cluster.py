# %% Packages
import time
from datetime import datetime
import sys
import random
from G2S_oldversion import *
# deep learning packages
import torch
from data_utils import *
from statistics import mean
import os
import numpy as np
import argparse
import pickle



# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

# Call data
parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original", default="original", choices=["augmented", "original"])
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="oldtok", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--beta", default=1, help="option: <any number>, schedule", choices=["normalVAE","schedule"])
parser.add_argument("--loss", default="ce", choices=["ce","wce"])
parser.add_argument("--AE_Warmup", default=False, action='store_true')
parser.add_argument("--seed", default=42)
parser.add_argument("--initialization", default="random", choices=["random", "xavier", "kaiming"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=0)
parser.add_argument("--dec_layers", type=int, default=4)



args = parser.parse_args()
seed = args.seed

augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "train"
data_augment = "old" # new or old
dict_train_loader = torch.load('dataloaders_'+data_augment+'augment/dict_'+dataset_type+'_loader_'+augment+'_'+tokenization+'.pt')


num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

# Load model
# Create an instance of the G2S model from checkpoint
filepath=os.path.join(os.getcwd(),'Checkpoints_new/Model_onlytorchseed_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/model_best_loss.pt')
if os.path.isfile(filepath):
    if args.ppguided:
        model_type = G2S_PPguided
    else: 
        model_type = G2S_Gab
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model_config = checkpoint["model_config"]
    batch_size = model_config['batch_size']
    hidden_dimension = model_config['hidden_dimension']
    embedding_dimension = model_config['embedding_dim']
    vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')
    if model_config['loss']=="wce":
        vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
        class_weights = token_weights(vocab_file)
        class_weights = torch.FloatTensor(class_weights)
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, loss_weights=class_weights, add_latent=add_latent)
    else: model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, add_latent=add_latent)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Directory to save results
    # Directory to save results
    dir_name= 'Checkpoints_new/Model_onlytorchseed_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f'Run over training set')

    #  Run over all epoch
    batches = list(range(len(dict_train_loader)))
    test_ce_losses = []
    test_total_losses = []
    test_kld_losses = []
    test_accs = []

    model.eval()
    model.beta = 1.0
    test_loss = 0
    # Iterate in batches over the training/test dataset.
    latents = []
    y1 = []
    y2 = []
    monomers = []
    stoichiometry = []
    connectivity_pattern = []
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if augment=='augmented' and dataset_type=='train': # otherwise it takes to long for train dataset
                if i>=500: 
                    break
            data = dict_train_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_train_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_train_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)

            # Perform a single forward pass.
            loss, recon_loss, kl_loss, acc, predictions, target, z = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            latents.append(z.cpu().numpy())
            y1.append(data.y1.cpu().numpy())
            y2.append(data.y2.cpu().numpy())
            monomers.append(data.monomer_smiles)
            targ_list = target.tolist()
            stoichiometry.extend(["|".join(combine_tokens(tokenids_to_vocab(targ_list_sub, vocab),tokenization=tokenization).split("|")[1:3]) for targ_list_sub in targ_list])
            connectivity_pattern.extend([combine_tokens(tokenids_to_vocab(targ_list_sub, vocab), tokenization=tokenization).split("|")[-1].split(':')[1] for targ_list_sub in targ_list])
            test_ce_losses.append(recon_loss.item())
            test_total_losses.append(loss.item())
            test_kld_losses.append(kl_loss.item())
            test_accs.append(acc.item())

    test_total = mean(test_total_losses)
    test_kld = mean(test_kld_losses)
    test_acc = mean(test_accs)
    latent_space = np.concatenate(latents, axis=0)
    y1_all = np.concatenate(y1, axis=0)
    y2_all = np.concatenate(y2, axis=0)
    with open(dir_name+'stoichiometry_'+dataset_type, 'wb') as f:
        pickle.dump(stoichiometry, f)
    with open(dir_name+'connectivity_'+dataset_type, 'wb') as f:
        pickle.dump(connectivity_pattern, f)
    with open(dir_name+'monomers_'+dataset_type, 'wb') as f:
        pickle.dump(monomers, f)
    with open(dir_name+'latent_space_'+dataset_type+'.npy', 'wb') as f:
        np.save(f, latent_space)
    with open(dir_name+'y1_all_'+dataset_type+'.npy', 'wb') as f:
        np.save(f, y1_all)
    with open(dir_name+'y2_all_'+dataset_type+'.npy', 'wb') as f:
        np.save(f, y2_all)

    print(f"Trainset: Total Loss: {test_total:.5f} | KLD: {test_kld:.5f} | ACC: {test_acc:.5f}")

    print(f'STARTING TEST')

    dataset_type = "test"
    data_augment = "old" # new or old
    dict_test_loader = torch.load('dataloaders_'+data_augment+'augment/dict_'+dataset_type+'_loader_'+augment+'_'+tokenization+'.pt')
    #  Run over all epoch
    batches = list(range(len(dict_test_loader)))
    test_ce_losses = []
    test_total_losses = []
    test_kld_losses = []
    test_accs = []

    model.eval()
    model.beta = 1.0
    test_loss = 0
    # Iterate in batches over the training/test dataset.
    latents = []
    y1 = []
    y2 = []
    monomers = []
    stoichiometry = []
    connectivity_pattern = []
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if augment=='augmented' and dataset_type=='train': # otherwise it takes to long for train dataset
                if i>=500: 
                    break
            data = dict_test_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_test_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)

            # Perform a single forward pass.
            loss, recon_loss, kl_loss, acc, predictions, target, z = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            latents.append(z.cpu().numpy())
            y1.append(data.y1.cpu().numpy())
            y2.append(data.y2.cpu().numpy())
            monomers.append(data.monomer_smiles)
            targ_list = target.tolist()
            stoichiometry.extend(["|".join(combine_tokens(tokenids_to_vocab(targ_list_sub, vocab),tokenization=tokenization).split("|")[1:3]) for targ_list_sub in targ_list])
            connectivity_pattern.extend([combine_tokens(tokenids_to_vocab(targ_list_sub, vocab), tokenization=tokenization).split("|")[-1].split(':')[1] for targ_list_sub in targ_list])
            test_ce_losses.append(recon_loss.item())
            test_total_losses.append(loss.item())
            test_kld_losses.append(kl_loss.item())
            test_accs.append(acc.item())

    test_total = mean(test_total_losses)
    test_kld = mean(test_kld_losses)
    test_acc = mean(test_accs)
    latent_space = np.concatenate(latents, axis=0)
    y1_all = np.concatenate(y1, axis=0)
    y2_all = np.concatenate(y2, axis=0)
    with open(dir_name+'stoichiometry_'+dataset_type, 'wb') as f:
        pickle.dump(stoichiometry, f)
    with open(dir_name+'connectivity_'+dataset_type, 'wb') as f:
        pickle.dump(connectivity_pattern, f)
    with open(dir_name+'monomers_'+dataset_type, 'wb') as f:
        pickle.dump(monomers, f)
    with open(dir_name+'latent_space_'+dataset_type+'.npy', 'wb') as f:
        np.save(f, latent_space)
    with open(dir_name+'y1_all_'+dataset_type+'.npy', 'wb') as f:
        np.save(f, y1_all)
    with open(dir_name+'y2_all_'+dataset_type+'.npy', 'wb') as f:
        np.save(f, y2_all)

    print(f"Testset: Total Loss: {test_total:.5f} | KLD: {test_kld:.5f} | ACC: {test_acc:.5f}")

else: print("The model training diverged and there are is no trained model file!")