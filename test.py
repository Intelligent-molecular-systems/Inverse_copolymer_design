# %% Packages
import time
from datetime import datetime
import sys
import random
from G2S import *
# deep learning packages
import torch
from data_utils import *
from statistics import mean
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import umap
from mpl_toolkits.mplot3d import Axes3D



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
dataset_type = "val"
augment = "augmented"
tokenization = "oldtok"

dict_test_loader = torch.load('dict_'+dataset_type+'_loader_'+augment+'_'+tokenization+'.pt')


num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features

# Load model
# Create an instance of the G2S model from checkpoint


checkpoint = torch.load(os.path.join(os.getcwd(),'Checkpoints/Model_h=300_decL=4_nheads=4_beta=schedule_loss=wce_augmented_oldtok_randomseed_newVAE.pt'), map_location=torch.device('cpu'))
model_config = checkpoint["model_config"]
batch_size = model_config['batch_size']
hidden_dimension = model_config['hidden_dimension']
embedding_dimension = model_config['embedding_dim']
vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')
if model_config['loss']=="wce":
    vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    class_weights = token_weights(vocab_file)
    class_weights = torch.FloatTensor(class_weights)
    model = G2S_Gab(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab, loss_weights=class_weights)
else: model = G2S_Gab(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab)
model.load_state_dict(checkpoint['model_state_dict'])
#model.load_state_dict(torch.load('Models_Weights/weights_training='+str(1)+'.pt', map_location=torch.device('cpu')))
model.to(device)

# Directory to save results
dir_name= 'Model_h='+str(model_config['hidden_dimension'])+'_decL='+str(model_config['decoder_num_layers'])+'_nheads='+str(model_config['num_attention_heads'])+'_beta='+str(model_config['beta'])+'_loss='+str(model_config['loss'])+'_augmented_oldtok_randomseed_newVAE'+'/'
if not os.path.exists(dir_name):
   os.makedirs(dir_name)

print(f'STARTING TEST')

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



### INFERENCE ###

if dataset_type=="val": 
    with torch.no_grad():
    # only for first batch
        model.eval()
        batch = 0
        data = dict_test_loader[str(batch)][0]
        data.to(device)
        dest_is_origin_matrix = dict_test_loader[str(batch)][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
        inc_edges_to_atom_matrix.to(device)
        predictions, _, _, z = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
        z_0_1 = torch.mean(z[0:2],0) # mean between the first two samples z in batch 
        z_rand = torch.randn(embedding_dimension)*0.01
        predictions_interp, _, _, z = model.inference(data=z_0_1, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
        predictions_rand, _, _, z = model.inference(data=z_rand, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
        vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')
        # save predicitons of first validation batch in text file
        prediction_strings = ["prediction"+str(sample+1)+": "+combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        real_strings = ["truth"+str(sample+1)+": "+combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab),tokenization=tokenization) for sample in range(len(data))]

        with open(dir_name+dataset_type+'inference'+'.txt', 'w+') as f:
            for i, s in enumerate(prediction_strings):
                f.write(s+'\n')
                f.write(real_strings[i]+'\n')
            f.write("Prediction from averaged z between samples 1 and 2: "+combine_tokens(tokenids_to_vocab(predictions_interp[0][0].tolist(), vocab), tokenization=tokenization))
            f.write("Prediction from random z: "+combine_tokens(tokenids_to_vocab(predictions_rand[0][0].tolist(), vocab), tokenization=tokenization))
        f.close()

        # Evaluation of validation set reconstruction accuracy (inference)
        #monomer_smiles_true = [poly_smiles.split("|")[0].split('.') for poly_smiles in real_strings] 
        #monomer_smiles_predicted = [poly_smiles.split("|")[0].split('.') for poly_smiles in prediction_strings]

        #monomer_weights_predicted = [poly_smiles.split("|")[1:-1] for poly_smiles in prediction_strings]

## interpolation between two samples z and decoding it 

# Start from here if the files are already created

with open(dir_name+'latent_space_'+dataset_type+'.npy', 'rb') as f:
    latent_space = np.load(f)
    print(latent_space.shape)
with open(dir_name+'y1_all_'+dataset_type+'.npy', 'rb') as f:
    y1_all = np.load(f)
with open(dir_name+'y2_all_'+dataset_type+'.npy', 'rb') as f:
    y2_all = np.load(f)
with open(dir_name+'monomers_'+dataset_type, "rb") as f:   # Unpickling
    monomers = pickle.load(f)
with open(dir_name+'stoichiometry_'+dataset_type, "rb") as f:   # Unpickling
    stoichiometry = pickle.load(f)
with open(dir_name+'connectivity_'+dataset_type, "rb") as f:   # Unpickling
    connectivity_pattern = pickle.load(f)


### Dimensionality reduction and plots ###
# Dimensionality reduction
dim_red_type = "umap"
#PCA
if dim_red_type == "pca":
    if dataset_type=="train":
        reducer = PCA(n_components=2).fit(latent_space)
        with open(dir_name+dim_red_type+'_fitted_'+dataset_type, 'wb') as f:
            pickle.dump(reducer, f)
        z_embedded = reducer.transform(latent_space)
    if dataset_type=="val":
        with open(dir_name+dim_red_type+'_fitted_train', 'rb') as f:
            reducer = pickle.load(f)
        z_embedded = reducer.transform(latent_space)
#pca = PCA(n_components=2)
#pca_latent = pca.fit_transform(latent_space)


# UMAP
if dim_red_type == "umap":
    if dataset_type=="train":
        reducer = umap.UMAP(n_components=2, min_dist=0.25).fit(latent_space)
        with open(dir_name+'umap_fitted_train_'+dataset_type, 'wb') as f:
            pickle.dump(reducer, f)
        z_embedded = reducer.embedding_
    if dataset_type=="val":
        with open(dir_name+'umap_fitted_train_train', 'rb') as f:
            reducer = pickle.load(f)
        z_embedded = reducer.transform(latent_space)

# TSNE
if dim_red_type == "tsne":
# model = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100)
# z_embedded = model.fit_transform(latent_space)
    if dataset_type=="train":
        reducer = TSNE(n_components=2).fit(latent_space)
        with open(dir_name+dim_red_type+'_fitted_'+dataset_type, 'wb') as f:
            pickle.dump(reducer, f)
        z_embedded = reducer.embedding_
    if dataset_type=="val":
        with open(dir_name+dim_red_type+'_fitted_train', 'rb') as f:
            reducer = pickle.load(f)
        z_embedded = reducer.fit_transform(latent_space)

# Get all the A monomers and create monomer label list samples x 1
monomersA = {}
sample_idx = 0 
for b in monomers:
    for sample in b: 
        monomersA[sample_idx] = sample[0]
        sample_idx+=1

unique_A_monomers = list(set(monomersA.values()))
print(unique_A_monomers)

# PLOTS
plt.figure(0)
plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=1, c=y1_all, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title('Electron affinity')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latenty1_'+dim_red_type+'.png')

plt.figure(1)
plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=1, c=y2_all, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title('Ionization potential')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latenty2_'+dim_red_type+'.png')

plt.figure(2)
# Color information; create custom colormap
# dictionary with A monomers
label_color_dict = {'[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2': '#e3342f',
                    '[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12': '#f6993f',
                    '[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2': '#ffed4a',
                    '[*:1]c1cc(F)c([*:2])cc1F': '#38c172',
                    '[*:1]c1ccc([*:2])cc1': '#4dc0b5',
                    '[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34': '#3490dc',
                    '[*:1]c1ccc(-c2ccc([*:2])s2)s1':"#6574cd" ,
                    '[*:1]c1cc2cc3sc([*:2])cc3cc2s1':'#9561e2',
                    '[*:1]c1ccc([*:2])c2nsnc12':'#f66d9b',
                    'no_A_monomer':'#808080'
                    }
if dataset_type=='val':
    label_color_dict = {'[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2': '#e3342f',
                    '[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12': '#f6993f',
                    '[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2': '#ffed4a',
                    '[*:1]c1cc(F)c([*:2])cc1F': '#38c172',
                    '[*:1]c1ccc([*:2])cc1': '#4dc0b5',
                    '[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34': '#3490dc',
                    '[*:1]c1ccc(-c2ccc([*:2])s2)s1':"#6574cd" ,
                    '[*:1]c1cc2cc3sc([*:2])cc3cc2s1':'#9561e2',
                    '[*:1]c1ccc([*:2])c2nsnc12':'#f66d9b',
                    }
all_labels = list(label_color_dict.keys())
all_colors = list(label_color_dict.values())
n_colors = len(all_colors)
cm = LinearSegmentedColormap.from_list('custom_colormap', all_colors, N=n_colors)

# Lables
labels = ('[*:1]c1cc(F)c([*:2])cc1F', '[*:1]c1cc2cc3sc([*:2])cc3cc2s1', '[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2', '[*:1]c1ccc(-c2ccc([*:2])s2)s1', '[*:1]c1ccc([*:2])cc1', '[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34', '[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2', '[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12', '[*:1]c1ccc([*:2])c2nsnc12', 'no_A_monomer')
if dataset_type=='val': labels = ('[*:1]c1cc(F)c([*:2])cc1F', '[*:1]c1cc2cc3sc([*:2])cc3cc2s1', '[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2', '[*:1]c1ccc(-c2ccc([*:2])s2)s1', '[*:1]c1ccc([*:2])cc1', '[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34', '[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2', '[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12', '[*:1]c1ccc([*:2])c2nsnc12')
# Get indices from color list for given labels
def assign_color(mon):
    try: 
        return all_colors.index(label_color_dict[mon]) 
    except:
        return all_colors.index(label_color_dict['no_A_monomer'])

color_idx = [assign_color(monomerA) for key,monomerA in monomersA.items()]

# Customize colorbar and plot
sc = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=2, c=color_idx, cmap=cm)
c_ticks = np.arange(n_colors) * (n_colors / (n_colors + 1)) + (2 / n_colors)
cbar = plt.colorbar(sc, ticks=c_ticks)
cbar.ax.set_yticklabels(all_labels)
cbar.ax.set_title('Monomer A type')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latent_Amonomers_'+dim_red_type+'.png', bbox_inches='tight')

plt.close()

# Stoichiometry
plt.figure(3)
label_color_dict = {'0.5|0.5': '#e3342f',
                    '0.25|0.75': '#f6993f',
                    '0.75|0.25': '#ffed4a'}
all_labels = list(label_color_dict.keys())
all_colors = list(label_color_dict.values())
n_colors = len(all_colors)
cm = LinearSegmentedColormap.from_list('custom_colormap', all_colors, N=n_colors)
labels = {'0.5|0.5':'1:1','0.25|0.75':'1:3','0.75|0.25':'3:1'}
all_labels=[labels[x] for x in all_labels]

# Get indices from color list for given labels
color_idx = [all_colors.index(label_color_dict[st]) for st in stoichiometry]

# Customize colorbar and plot
sc = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=2, c=color_idx, cmap=cm)
c_ticks = np.arange(n_colors) * (n_colors / (n_colors + 1)) + (1 / (n_colors+1))
cbar = plt.colorbar(sc, ticks=c_ticks)
cbar.ax.set_yticklabels(all_labels)
cbar.ax.set_title('Stoichiometric ratio')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latent_stoichimetry_'+dim_red_type+'.png', bbox_inches='tight')
plt.close()

# Connectivity
plt.figure(3)
label_color_dict = {'0.5': '#e3342f',
                    '0.375': '#f6993f',
                    '0.25': '#ffed4a'}
all_labels = list(label_color_dict.keys())
all_colors = list(label_color_dict.values())
n_colors = len(all_colors)
cm = LinearSegmentedColormap.from_list('custom_colormap', all_colors, N=n_colors)
labels = {'0.5':'Alternating','0.25':'Random','0.375':'Block'}
all_labels=[labels[x] for x in all_labels]


# Get indices from color list for given labels
color_idx = [all_colors.index(label_color_dict[st]) for st in connectivity_pattern]

# Customize colorbar and plot
sc = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=2, c=color_idx, cmap=cm)
c_ticks = np.arange(n_colors) * (n_colors / (n_colors + 1)) + (1 / (n_colors+1))
cbar = plt.colorbar(sc, ticks=c_ticks)
cbar.ax.set_yticklabels(all_labels)
cbar.ax.set_title('Chain architecture')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latent_connectivity_'+dim_red_type+'.png', bbox_inches='tight')


print('Done!\n')


# %%