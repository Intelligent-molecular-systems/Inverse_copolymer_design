# %% Packages
import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)
from data_processing.data_utils import *
from data_processing.rdkit_poly import *
from model.G2S_clean import *

from model.G2S_clean import *
from data_processing.data_utils import *
from data_processing.Function_Featurization_Own import poly_smiles_to_graph


import time
from datetime import datetime
import random
# deep learning packages
from torch_geometric.loader import DataLoader
import torch
from statistics import mean
import numpy as np
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap
import pickle
#import umap
from mpl_toolkits.mplot3d import Axes3D
import argparse
#from sklearn.neighbors import KernelDensity



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
parser.add_argument("--max_alpha", type=float, default=0.1)
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
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
filepath = os.path.join(main_dir_path,'Checkpoints/', model_name,"model_best_loss.pt")
dir_name = os.path.join(main_dir_path,'Checkpoints/', model_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Start from here if the files are already created
 
with open(dir_name+'latent_space_'+dataset_type+'.npy', 'rb') as f:
    latent_space = np.load(f)
    print(latent_space.shape)



""" 
# Define bandwidth (bandwidth controls the smoothness of the kernel density estimate)
bandwidth = 0.1

plt.figure(1)
colors = [plt.cm.jet(random.random()) for _ in range(latent_space.shape[1])]

for i in range(latent_space.shape[1]):

    # Fit kernel density estimator for latent dimension i
    kde_i = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_i.fit(latent_space[:,i].reshape(-1, 1))

    # Create a range of values for the x-axis
    x_values = np.linspace(np.min(latent_space),np.max(latent_space), 1000)
    # Evaluate the KDE on the range of values
    density_i = np.exp(kde_i.score_samples(x_values.reshape(-1, 1)))

    # Plotting
    plt.plot(x_values, density_i, color=colors[i], label=f'Dimension {i+1}')
plt.xlabel('value')
plt.ylabel('Density')
plt.title('Kernel Density Estimation (Electron affinity)')
plt.legend()
plt.show()
plt.savefig(dir_name+'latentdims_kde.png') """


# Plot EA vs IP
with open(dir_name+'y1_all_'+dataset_type+'.npy', 'rb') as f:
    y1_all = np.load(f)
with open(dir_name+'y2_all_'+dataset_type+'.npy', 'rb') as f:
    y2_all = np.load(f)

with open(dir_name+'yp_all_'+dataset_type+'.npy', 'rb') as f:
    yp_all = np.load(f)
yp1_all = [yp[0] for yp in yp_all]
yp2_all = [yp[1] for yp in yp_all]


# PLOTS
plt.figure(2)
font = {'size'   : 18}
import matplotlib
matplotlib.rc('font', **font)
# Convert lists to numpy arrays

array1 = np.array(y1_all)
array2 = np.array(y2_all)
# Create a boolean mask identifying NaN values
nan_mask = np.isnan(array1)
non_nan_mask = ~nan_mask

# Use the mask to filter the array
array1 = array1[non_nan_mask]
array2 = array2[non_nan_mask]

# Calculate the objective values
#obj_vals = array1 + np.abs(array2 - 1.0)#
obj_vals = np.abs(array1 + 2.0) +  np.abs(array2 - 1.2)

print(obj_vals)
# Create the scatter plot
plt.scatter(array1, array2, c=obj_vals, cmap='inferno', s=1)

cwd = os.getcwd()

# Add a colorbar
plt.colorbar(label=r'Objective: $\mathit{f}(\mathbf{z})$')
plt.title(r'$\mathit{IP}$ vs. $\mathit{EA}$ (training data)')
plt.xlabel(r'$\mathit{EA}$ in eV')
plt.ylabel(r'$\mathit{IP}$ in eV')
plt.subplots_adjust(bottom=0.15)
plt.savefig(cwd+'IPvsEA_mimickpeak.png')

# What are the best objective values
# Number of largest elements to print
N = 10

# Get the indices that would sort the array in descending order
sorted_indices = np.argsort(obj_vals)


# Select the top N elements using the sorted indices
smallest_elements = obj_vals[sorted_indices][:N]
print(smallest_elements, np.mean(smallest_elements))


plt.figure(3)
plt.scatter(yp1_all, yp2_all, s=1)
plt.title('IP vs. EA (augmented data)')
plt.xlabel("EA in eV")
plt.ylabel("IP in eV")
plt.savefig(cwd+'IPvsEA_mimickpeak_pred.png')
""" 
with open(dir_name+'monomers_'+dataset_type, "rb") as f:   # Unpickling
    monomers = pickle.load(f)
# Get all the A monomers and create monomer label list samples x 1
monomersA = {}
sample_idx = 0 
for b in monomers:
    for sample in b: 
        monomersA[sample_idx] = sample[0]
        sample_idx+=1

unique_A_monomers = list(set(monomersA.values()))
print(unique_A_monomers)
print(len(unique_A_monomers))

graph_list = torch.load(main_dir_path+'/data/Graphs_list_augmented_RT_tokenized.pt')

monomer_combs = []
monomers = []

for i, data in enumerate(graph_list):
    if augment=="augmented_canonical":
        monomers.append(data.monomer_smiles_nocan[0])
        monomer_combs.append(".".join(data.monomer_smiles_nocan))

    else: 
        monomers.append(data.monomer_smiles[0])
        monomer_combs.append("".join(data.monomer_smiles))

sample_idx = 0 
unique_A_monomers = list(set(monomers))
unique_monomers_combs = list(set(monomer_combs))
print(len(unique_monomers_combs))
print(unique_A_monomers)
print(len(unique_A_monomers))


dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_augmented_RT_tokenized.pt')
batches = list(range(len(dict_train_loader)))

monomer_combs = []
monomers = []

for i, batch in enumerate(batches):
    data = dict_train_loader[str(batch)][0]
    if augment=="augmented_canonical":
        monomers.append(data.monomer_smiles_nocan[0])
        monomer_combs.append(".".join(data.monomer_smiles_nocan))

    else: 
        monomers.append(data.monomer_smiles[0])
        #monomer_combs.append("".join(data.monomer_smiles))

# Get all the A monomers and create monomer label list samples x 1
monomersA = []
sample_idx = 0 
for s in monomers:
    monomersA.append(s[0])

unique_A_monomers = list(set(monomersA))
print(unique_A_monomers)
print(len(unique_A_monomers)) """

