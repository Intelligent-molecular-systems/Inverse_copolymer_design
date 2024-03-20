# %% Packages
import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)
from data_processing.data_utils import *
from data_processing.rdkit_poly import *
from model.G2S_clean import *

import time
from datetime import datetime
import random
# deep learning packages
import torch
from statistics import mean
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import umap
from mpl_toolkits.mplot3d import Axes3D
import argparse


# setting device on GPU if available, else CPU
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

# Call data
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
parser.add_argument("--dim_red_type", default="pca", choices=["umap", "pca", "tsne"])


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


vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
vocab = load_vocab(vocab_file=vocab_file)

# Directory to save results
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
dir_name = os.path.join(main_dir_path,'Checkpoints/', model_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

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
with open(dir_name+'yp_all_'+dataset_type+'.npy', 'rb') as f:
    yp_all = np.load(f)
yp1_all = [yp[0] for yp in yp_all]
yp2_all = [yp[1] for yp in yp_all]

### Dimensionality reduction and plots ###
# Dimensionality reduction
dim_red_type = args.dim_red_type
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
        reducer = umap.UMAP(n_components=2, min_dist=0.5).fit(latent_space)
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
plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=1, c=yp1_all, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title('Electron affinity')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latentyp1_'+dim_red_type+'.png')

plt.figure(3)
plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=1, c=yp2_all, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title('Ionization potential')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latentyp2_'+dim_red_type+'.png')

plt.figure(4)
# Color information; create custom colormap
# dictionary with A monomers

amons=['[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2','[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12','[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2','[*:1]c1cc(F)c([*:2])cc1F','[*:1]c1ccc([*:2])cc1','[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34', '[*:1]c1ccc(-c2ccc([*:2])s2)s1','[*:1]c1cc2cc3sc([*:2])cc3cc2s1','[*:1]c1ccc([*:2])c2nsnc12', 'no_A_monomer']
cols=['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc',"#6574cd" ,'#9561e2','#f66d9b','#808080']
label_color_dict = {amons[i]:cols[i] for i in range(len(amons))}
if dataset_type=='test':
    label_color_dict = {amons[i]:cols[i] for i in range(len(amons)-1)}
all_labels = list(label_color_dict.keys())
all_colors = list(label_color_dict.values())
n_colors = len(all_colors)
cm = LinearSegmentedColormap.from_list('custom_colormap', all_colors, N=n_colors)

# Lables
labels = amons
# Get indices from color list for given labels
def assign_color(mon):
    try: 
        alpha = 1.0
        color = all_colors.index(label_color_dict[mon]) 
        return color, alpha
    except:
        alpha = 0.1
        color = all_colors.index(label_color_dict['no_A_monomer'])
        return color, alpha

color_idx = [assign_color(monomerA)[0] for key,monomerA in monomersA.items()]
alphas = [assign_color(monomerA)[1] for key,monomerA in monomersA.items()]

# Customize colorbar and plot
sc = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=2, c=color_idx, cmap=cm, alpha=alphas)
c_ticks = np.arange(n_colors) * (n_colors / (n_colors + 1)) + (2 / n_colors)
cbar = plt.colorbar(sc, ticks=c_ticks)
cbar.ax.set_yticklabels(all_labels)
cbar.ax.set_title('Monomer A type')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latent_Amonomers_'+dim_red_type+'.png', bbox_inches='tight')

plt.close()

# Stoichiometry
plt.figure(5)
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
plt.figure(6)
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


print('Now for validation set!\n')

dataset_type = "test"

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
with open(dir_name+'yp_all_'+dataset_type+'.npy', 'rb') as f:
    yp_all = np.load(f)
yp1_all = [yp[0] for yp in yp_all]
yp2_all = [yp[1] for yp in yp_all]


### Dimensionality reduction and plots ###
# Dimensionality reduction
#PCA
if dim_red_type == "pca":
    if dataset_type=="train":
        reducer = PCA(n_components=2).fit(latent_space)
        with open(dir_name+dim_red_type+'_fitted_'+dataset_type, 'wb') as f:
            pickle.dump(reducer, f)
        z_embedded = reducer.transform(latent_space)
    if dataset_type=="test":
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
    if dataset_type=="test":
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
    if dataset_type=="test":
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
plt.figure(7)
plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=1, c=y1_all, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title('Electron affinity')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latenty1_'+dim_red_type+'.png')

plt.figure(8)
plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=1, c=y2_all, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title('Ionization potential')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latenty2_'+dim_red_type+'.png')

plt.figure(9)
# Color information; create custom colormap
# dictionary with A monomers
amons=['[*:1]c1ccc2c(c1)C(C)(C)c1cc([*:2])ccc1-2','[*:1]c1ccc2c(c1)[nH]c1cc([*:2])ccc12','[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2','[*:1]c1cc(F)c([*:2])cc1F','[*:1]c1ccc([*:2])cc1','[*:1]c1cc2ccc3cc([*:2])cc4ccc(c1)c2c34', '[*:1]c1ccc(-c2ccc([*:2])s2)s1','[*:1]c1cc2cc3sc([*:2])cc3cc2s1','[*:1]c1ccc([*:2])c2nsnc12', 'no_A_monomer']
cols=['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc',"#6574cd" ,'#9561e2','#f66d9b','#808080']
label_color_dict = {amons[i]:cols[i] for i in range(len(amons))}

all_labels = list(label_color_dict.keys())
all_colors = list(label_color_dict.values())
n_colors = len(all_colors)
cm = LinearSegmentedColormap.from_list('custom_colormap', all_colors, N=n_colors)

# Lables
labels = amons
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
plt.figure(10)
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
plt.figure(11)
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

plt.figure(12)
plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=1, c=yp1_all, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title('Electron affinity')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latentyp1_'+dim_red_type+'.png')

plt.figure(13)
plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=1, c=yp2_all, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title('Ionization potential')
plt.xlabel(dim_red_type+" 1")
plt.ylabel(dim_red_type+" 2")
plt.savefig(dir_name+dataset_type+'_latentyp2_'+dim_red_type+'.png')
plt.close()

print('Done!\n')


# %%


# %%