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
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import pickle
import umap
from mpl_toolkits.mplot3d import Axes3D
import argparse


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


dataset_type = "val" 
data_augment = "old" # new or old
dict_train_loader = torch.load('dataloaders_'+data_augment+'augment/dict_test_loader_'+augment+'_'+tokenization+'.pt')

vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')

# Directory to save results
dir_name= 'Checkpoints_new/Model_onlytorchseed_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)



# Start from here if the files are already created
torch.manual_seed(42)
z_novel=[]
for i in range(250):
    z_rand = torch.randn((64,32))
    z_novel.append(z_rand.numpy())
novel_latent_space = np.concatenate(z_novel, axis=0)
std=0.025
with open(dir_name+'seed_polymers_latents_noise'+str(std)+'.npy', 'rb') as f:
    seed_samples_latents = np.load(f)
    seed_samples_latents=seed_samples_latents.reshape(-1, seed_samples_latents.shape[2])
with open(dir_name+'seed_polymer_z.npy', 'rb') as f:
    seed_latent = np.load(f)
    seed_latent = seed_latent.reshape(1, -1)
with open(dir_name+'latent_space_train.npy', 'rb') as f:
    train_latent_space = np.load(f)
    print(train_latent_space.shape)
with open(dir_name+'monomers_train', "rb") as f:   # Unpickling
    monomers = pickle.load(f)
with open(dir_name+'generated_polymers_from_seed_noise'+str(std)+'_metrics.pkl', "rb") as f:   # Unpickling
    seed_metrics = pickle.load(f)


with open(dir_name+'umap_fitted_train_train', 'rb') as f:
    reducer = pickle.load(f)
z_embedded_train = reducer.transform(train_latent_space)
z_embedded_seed = reducer.transform(seed_latent)
print(seed_samples_latents.shape, train_latent_space.shape)
z_embedded_seed_samples = reducer.transform(seed_samples_latents)
#print(z_embedded_novel.shape, z_embedded_train.shape)

# Get all the A monomers and create monomer label list samples x 1
monomersA = {}
sample_idx = 0 
for b in monomers:
    for sample in b: 
        monomersA[sample_idx] = sample[0]
        sample_idx+=1

unique_A_monomers = list(set(monomersA.values()))


plt.figure(0)
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
                    'no_A_monomer':'#808080'
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
sc = plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=color_idx, cmap=cm, alpha=.3)
plt.scatter(z_embedded_seed_samples[:, 0], z_embedded_seed_samples[:, 1], s=1)
plt.scatter(z_embedded_seed_samples[:, 0], z_embedded_seed_samples[:, 1], s=15, marker='*', c='brown')
plt.scatter(z_embedded_seed[:,0], z_embedded_seed[:, 1], s=30, marker='x',c='black')
#plt.text(z_embedded_seed[:,0] + 0.1, z_embedded_seed[:, 1] + 0.1, f'Seed', fontsize=12)
c_ticks = np.arange(n_colors) * (n_colors / (n_colors + 1)) + (2 / n_colors)
cbar = plt.colorbar(sc, ticks=c_ticks)
cbar.ax.set_yticklabels(all_labels)
cbar.ax.set_title('Monomer A type')
plt.xlabel("umap1")
plt.ylabel("umap2")
plt.savefig(dir_name+'novel_pols_'+str(std)+'_in_trainspace.png', bbox_inches='tight')

plt.close()
# %%
# Additional plot:
# Data
categories = ["Mon. A", "Mon. B ", "Stoich.", "Chain arch."]
num_predictions = [seed_metrics["isn_a"],seed_metrics["isn_b"],seed_metrics["isn_s"],seed_metrics["isn_c"]]  # Replace with your actual data

# Create a bar plot
# Data for stacked bars (two colors)
num_predictions_stacked = [(seed_metrics["isn_a"]-seed_metrics["novelA"], seed_metrics["novelA"]), (seed_metrics["isn_b"]-seed_metrics["novelB"], seed_metrics["novelB"]), (seed_metrics["isn_s"],0), (seed_metrics["isn_c"], 0)]  # Replace with your actual stacked data

# Create a bar plot with thinner bars (adjust the width as needed)
fig, ax = plt.subplots(figsize=(4, 6))
#matplotlib.rcParams.update({'font.size': 22})

for i, (bottom, top) in enumerate(num_predictions_stacked):
    if i==0:
        ax.bar(categories[i], bottom, width=0.4, label="in training data", color="grey")
        ax.bar(categories[i], top, width=0.4, bottom=bottom, label="novel", color='r', hatch='//')
    else: 
        ax.bar(categories[i], bottom, width=0.4, color="grey")
        ax.bar(categories[i], top, width=0.4,bottom=bottom, color='r', hatch='/')


# Add a legend
ax.legend(loc="upper right")
# Set labels and title
plt.ylabel("Number of samples")
plt.xlabel("Variation from seed molecule")

# Display the plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Ensure all labels fit in the plot
plt.savefig(dir_name+'novel_pols_'+str(std)+'_changes_barplot.png', bbox_inches='tight')
# %%