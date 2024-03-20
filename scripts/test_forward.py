# %% Packages
import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *

import time
from datetime import datetime
import sys
import random
# deep learning packages
import torch
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

parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original, augmented_canonical", default="original", choices=["augmented", "original", "augmented_canonical", "augmented_enum"])
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="oldtok", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32)
parser.add_argument("--beta", default=1, help="option: <any number>, schedule", choices=["normalVAE","schedule"])
parser.add_argument("--loss", default="ce", choices=["ce","wce"])
parser.add_argument("--AE_Warmup", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initialization", default="random", choices=["random", "xavier", "kaiming"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=0)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.01)
parser.add_argument("--epsilon", type=float, default=1)



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
dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')


num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

# Load model
# Create an instance of the G2S model from checkpoint
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
filepath = os.path.join(main_dir_path,'Checkpoints/', model_name,"model_best_loss.pt")
if os.path.isfile(filepath):
    if args.ppguided:
        model_type = G2S_VAE_PPguided
    else: 
        model_type = G2S_VAE
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model_config = checkpoint["model_config"]
    batch_size = model_config['batch_size']
    hidden_dimension = model_config['hidden_dimension']
    embedding_dimension = model_config['embedding_dim']
    vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    vocab = load_vocab(vocab_file=vocab_file)
    if model_config['loss']=="wce":
        class_weights = token_weights(vocab_file)
        class_weights = torch.FloatTensor(class_weights)
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, loss_weights=class_weights, add_latent=add_latent)
    else: model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, add_latent=add_latent)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Directory to save results
    dir_name=  os.path.join(main_dir_path,'Checkpoints/', model_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f'Run over training set')

    #  Run over all epoch
    batches = list(range(len(dict_train_loader)))
    test_ce_losses = []
    test_total_losses = []
    test_kld_losses = []
    test_accs = []
    test_mses = []

    model.eval()
    model.beta = 1.0
    test_loss = 0
    # Iterate in batches over the training/test dataset.
    latents = []
    y1 = []
    y2 = []
    y_p = []
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
            loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y_pred = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            latents.append(z.cpu().numpy())
            y_p.append(y_pred.cpu().numpy())
            y1.append(data.y1.cpu().numpy())
            y2.append(data.y2.cpu().numpy())
            if augment=="augmented_canonical":
                monomers.append(data.monomer_smiles_nocan)
            else: 
                monomers.append(data.monomer_smiles)
            targ_list = target.tolist()
            stoichiometry.extend(["|".join(combine_tokens(tokenids_to_vocab(targ_list_sub, vocab),tokenization=tokenization).split("|")[1:3]) for targ_list_sub in targ_list])
            connectivity_pattern.extend([combine_tokens(tokenids_to_vocab(targ_list_sub, vocab), tokenization=tokenization).split("|")[-1].split(':')[1] for targ_list_sub in targ_list])
            test_ce_losses.append(recon_loss.item())
            test_total_losses.append(loss.item())
            test_kld_losses.append(kl_loss.item())
            test_accs.append(acc.item())
            test_mses.append(mse.item())

    test_total = mean(test_total_losses)
    test_kld = mean(test_kld_losses)
    test_acc = mean(test_accs)

    latent_space = np.concatenate(latents, axis=0)
    y1_all = np.concatenate(y1, axis=0)
    y2_all = np.concatenate(y2, axis=0)
    y_p_all = np.concatenate(y_p, axis=0)
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
    with open(dir_name+'yp_all_'+dataset_type+'.npy', 'wb') as f:
        np.save(f, y_p_all)

    print(f"Trainset: Total Loss: {test_total:.5f} | KLD: {test_kld:.5f} | ACC: {test_acc:.5f}")

    print(f'STARTING TEST')

    dataset_type = "test"
    data_augment = "old" # new or old
    dict_test_loader = torch.load(main_dir_path+'/data/dict_test_loader_'+augment+'_'+tokenization+'.pt')
    #  Run over all epoch
    batches = list(range(len(dict_test_loader)))
    test_ce_losses = []
    test_total_losses = []
    test_kld_losses = []
    test_accs = []
    test_mses = []

    model.eval()
    model.beta = 1.0
    test_loss = 0
    # Iterate in batches over the training/test dataset.
    latents = []
    y1 = []
    y2 = []
    y_p = []
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
            loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y_pred = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
            latents.append(z.cpu().numpy())
            y_p.append(y_pred.cpu().numpy())
            y1.append(data.y1.cpu().numpy())
            y2.append(data.y2.cpu().numpy())
            if augment=="augmented_canonical":
                monomers.append(data.monomer_smiles_nocan)
            else: 
                monomers.append(data.monomer_smiles)
            targ_list = target.tolist()
            stoichiometry.extend(["|".join(combine_tokens(tokenids_to_vocab(targ_list_sub, vocab),tokenization=tokenization).split("|")[1:3]) for targ_list_sub in targ_list])
            connectivity_pattern.extend([combine_tokens(tokenids_to_vocab(targ_list_sub, vocab), tokenization=tokenization).split("|")[-1].split(':')[1] for targ_list_sub in targ_list])
            test_ce_losses.append(recon_loss.item())
            test_total_losses.append(loss.item())
            test_kld_losses.append(kl_loss.item())
            test_accs.append(acc.item())
            test_mses.append(mse.item())

    test_total = mean(test_total_losses)
    test_kld = mean(test_kld_losses)
    test_acc = mean(test_accs)

    latent_space = np.concatenate(latents, axis=0)
    y1_all = np.concatenate(y1, axis=0)
    y2_all = np.concatenate(y2, axis=0)
    y_p_all = np.concatenate(y_p, axis=0)
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
    with open(dir_name+'yp_all_'+dataset_type+'.npy', 'wb') as f:
        np.save(f, y_p_all)


    print(f"Testset: Total Loss: {test_total:.5f} | KLD: {test_kld:.5f} | ACC: {test_acc:.5f}")

else: print("The model training diverged and there are is no trained model file!")