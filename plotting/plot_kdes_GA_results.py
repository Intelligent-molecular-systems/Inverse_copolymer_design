import pickle

import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *
import matplotlib.pyplot as plt
import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original", default="augmented", choices=["augmented", "original"])
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="oldtok", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32)
parser.add_argument("--beta", default=1, help="option: <any number>, schedule", choices=["normalVAE","schedule"])
parser.add_argument("--loss", default="ce", choices=["ce","wce"])
parser.add_argument("--AE_Warmup", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initialization", default="random", choices=["random"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=0)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.1)
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
dir_name=  os.path.join(main_dir_path,'Checkpoints/', model_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

else: print("There is no trained model file!", dir_name)



objective_type="EAmin"
with open(dir_name+'optimization_results_custom_GA_correct_'+str(objective_type)+'.pkl', 'rb') as f:
    results_custom = pickle.load(f)

# Calculate distances between the BO and reencoded latents
Latents_RE = []
pred_RE = []
decoded_mols= []
pred_RE_corrected = []

for idx, (pop, res) in enumerate(list(results_custom.items())):
    population= int(pop)
    # loop through population
    pop_size = len(list(res["objective"]))
    for point in range(pop_size):
        L_re=res["latents_reencoded"][point]
        Latents_RE.append(L_re)
        pred_RE.append(res["predictions"][point])
        pred_RE_corrected.append(res["predictions_doublecorrect"][point])
        decoded_mols.append(res["string_decoded"][point])

iterations = range(len(pred_RE))
EA_re= [arr[0].cpu() for arr in pred_RE]
IP_re = [arr[1].cpu() for arr in pred_RE]
EA_re_c= [arr[0] for arr in pred_RE_corrected]
IP_re_c = [arr[1] for arr in pred_RE_corrected]

def top_n_molecule_indices(objective_values, decoded_mols, n_idx=10):
    # Get the indices of 20 molecules with the best objective values
    # Pair each value with its index
    # Filter out NaN values and keep track of original indices
    filtered_indexed_values = [(index, value) for index, value in enumerate(objective_values) if not math.isnan(value)]
    # Sort the indexed values by the value in ascending order and take n_idx best ones
    sorted_filtered_indexed_values = sorted(filtered_indexed_values, key=lambda x: x[1], reverse=False)
    _best_mols = []
    best_mols_count = {}
    top_idxs = []
    for index, value in sorted_filtered_indexed_values: 
        if not decoded_mols[index] in _best_mols: 
            top_idxs.append(index)
            best_mols_count[decoded_mols[index]]=1
            _best_mols.append(decoded_mols[index])
        else:
            best_mols_count[decoded_mols[index]]+=1
        if len(top_idxs)==20:
            break

    return top_idxs, best_mols_count


# TODO Implement extracting the right properties to fill all_y_p  in the code below. 

# Extract data for the curves
if objective_type=='mimick_peak':
    objective_values = [(np.abs(arr.cpu()[0]+2)+np.abs(arr.cpu()[1]-1.2)) for arr in pred_RE]
    objective_values_c = [(np.abs(arr[0]+2)+np.abs(arr[1]-1.2)) for arr in pred_RE_corrected]
elif objective_type=='mimick_best':
    objective_values = [(np.abs(arr.cpu()[0]+2.64)+np.abs(arr.cpu()[1]-1.61)) for arr in pred_RE]
    objective_values_c = [(np.abs(arr[0]+2.64)+np.abs(arr[1]-1.61)) for arr in pred_RE_corrected]
elif objective_type=='EAmin': 
    objective_values = [arr.cpu()[0]+np.abs(arr.cpu()[1]-1) for arr in pred_RE]
    objective_values_c = [arr[0]+np.abs(arr[1]-1) for arr in pred_RE_corrected]
elif objective_type =='max_gap':
    objective_values = [arr.cpu()[0]-arr.cpu()[1] for arr in pred_RE]
    objective_values_c = [arr[0]-arr[1] for arr in pred_RE_corrected]


top_20_indices, top_20_mols = top_n_molecule_indices(objective_values, decoded_mols, n_idx=500)
best_mols_t20 = {i+1: decoded_mols[i] for i in top_20_indices}
best_IPs = [IP_re[i] for i in top_20_indices]
best_EAs = [EA_re[i] for i in top_20_indices]
best_objs_t20 = {i+1: objective_values[i] for i in top_20_indices}

from sklearn.neighbors import KernelDensity

# Properties from dataset (real and augmented)
with open(dir_name+'y1_all_'+dataset_type+'.npy', 'rb') as f:
    y1_all = np.load(f)
with open(dir_name+'y2_all_'+dataset_type+'.npy', 'rb') as f:
    y2_all = np.load(f)
with open(dir_name+'yp_all_'+dataset_type+'.npy', 'rb') as f:
    yp_all = np.load(f)

y1_all=list(y1_all)
y2_all=list(y2_all)
yp1_all = [yp[0] for yp in yp_all]
yp2_all = [yp[1] for yp in yp_all]
# Do a KDE to check if the distributions of properties are similar (predicted vs. real lables)
""" y1 """
plt.figure(1)
real_distribution = np.array([r for r in y1_all if not np.isnan(r)])
augmented_distribution = np.array([p for p in yp1_all])
GA_distribution = np.array(best_EAs)


# Reshape the data
real_distribution = real_distribution.reshape(-1, 1)
augmented_distribution = augmented_distribution.reshape(-1, 1)
GA_distribution = GA_distribution.reshape(-1, 1)

# Define bandwidth (bandwidth controls the smoothness of the kernel density estimate)
bandwidth = 0.1

# Fit kernel density estimator for real data
kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_real.fit(real_distribution)
# Fit kernel density estimator for augmented data
kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_augmented.fit(augmented_distribution)
# Fit kernel density estimator for sampled data
kde_sampled_seed = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_sampled_seed.fit(GA_distribution)

# Create a range of values for the x-axis with some padding
padding = 0.5
x_min = min(np.min(real_distribution), np.min(augmented_distribution), np.min(GA_distribution)) - padding
x_max = max(np.max(real_distribution), np.max(augmented_distribution), np.max(GA_distribution)) + padding
x_values = np.linspace(x_min, x_max, 1000)
#x_values = np.linspace(min(np.min(real_distribution), np.min(augmented_distribution), np.min(GA_distribution)), max(np.max(real_distribution), np.max(augmented_distribution), np.max(GA_distribution)), 1000)
# Evaluate the KDE on the range of values
real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
GA_density = np.exp(kde_sampled_seed.score_samples(x_values.reshape(-1, 1)))

# Plotting
plt.rcParams.update({'font.size': 18, 'fontname':'Droid Sans'})  # Apply to all text elements

plt.plot(x_values, real_density, label='Real Data')
plt.fill_between(x_values, real_density, alpha=0.5)
plt.plot(x_values, augmented_density, label='Augmented Data')
plt.fill_between(x_values, augmented_density, alpha=0.5)
plt.plot(x_values, GA_density, label='Best 500 molecules GA')
plt.fill_between(x_values, GA_density, alpha=0.5)

plt.xlabel('EA (eV)')
plt.ylabel('Density')
#plt.title('Kernel Density Estimation (Electron affinity)')
plt.legend()
plt.show()

plt.savefig(dir_name+'KDE_EA_GA.png', dpi=300, bbox_inches='tight')

""" y2 """
plt.figure(2)
real_distribution = np.array([r for r in y2_all if not np.isnan(r)])
augmented_distribution = np.array([p for p in yp2_all])
GA_distribution = np.array(best_IPs)


# Reshape the data
real_distribution = real_distribution.reshape(-1, 1)
augmented_distribution = augmented_distribution.reshape(-1, 1)
GA_distribution = GA_distribution.reshape(-1, 1)

# Define bandwidth (bandwidth controls the smoothness of the kernel density estimate)
bandwidth = 0.1

# Fit kernel density estimator for real data
kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_real.fit(real_distribution)
# Fit kernel density estimator for augmented data
kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_augmented.fit(augmented_distribution)
# Fit kernel density estimator for sampled data
kde_sampled_seed = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_sampled_seed.fit(GA_distribution)

# Create a range of values for the x-axis
x_values = np.linspace(min(np.min(real_distribution), np.min(augmented_distribution), np.min(GA_distribution)), max(np.max(real_distribution), np.max(augmented_distribution), np.max(GA_distribution)), 1000)
# Evaluate the KDE on the range of values
real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
GA_density = np.exp(kde_sampled_seed.score_samples(x_values.reshape(-1, 1)))

# Plotting

plt.plot(x_values, real_density, label='Real Data')
plt.fill_between(x_values, real_density, alpha=0.5)
plt.plot(x_values, augmented_density, label='Augmented Data')
plt.fill_between(x_values, augmented_density, alpha=0.5)
plt.plot(x_values, GA_density, label='Best 500 molecules GA')
plt.fill_between(x_values, GA_density, alpha=0.5)

plt.xlabel('IP (eV)')
plt.ylabel('Density')
#plt.title('Kernel Density Estimation (Ionization potential)')
plt.legend()
plt.show()
plt.savefig(dir_name+'KDE_IP_GA.png', dpi=300, bbox_inches='tight')


# Histogram:
plt.figure(3)
plt.figure(figsize=(10, 6))

# Histogram for real data
plt.hist(real_distribution, bins=30, alpha=0.5, label='Real Data', density=True, edgecolor='k')
# Histogram for augmented data
plt.hist(augmented_distribution, bins=30, alpha=0.5, label='Augmented Data', density=True, edgecolor='k')
# Histogram for GA data
plt.hist(GA_distribution, bins=30, alpha=0.5, label='Best 500 molecules GA', density=True, edgecolor='k')

plt.xlabel('IP (eV)')
plt.ylabel('Density')
#plt.title('Histogram of Ionization Potential')
plt.legend()
plt.show()
plt.savefig(dir_name + 'Histogram_IP_GA.png', bbox_inches='tight')

