import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

import numpy as np
from optimization_custom.bayesian_optimization import BayesianOptimization
from bayes_opt.util import UtilityFunction
from bayes_opt.event import DEFAULT_EVENTS, Events

import argparse
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import pickle
import json
import pandas as pd
import math

from data_processing.data_utils import *
from data_processing.rdkit_poly import *
from data_processing.Smiles_enum_canon import SmilesEnumCanon

from model.G2S_clean import *
from data_processing.data_utils import *
from data_processing.Function_Featurization_Own import poly_smiles_to_graph
from data_processing.rdkit_poly import make_polymer_mol

import time

# setting device on GPU if available, else CPU
# setting device on GPU if available, else CPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original", default="augmented", choices=["augmented", "original"])
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="RT_tokenized", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32)
parser.add_argument("--beta", default="schedule", help="option: <any number>, schedule", choices=["normalVAE","schedule"])
parser.add_argument("--loss", default="wce", choices=["ce","wce"])
parser.add_argument("--AE_Warmup", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--initialization", default="random", choices=["random"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=1)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.0004)
parser.add_argument("--max_alpha", type=float, default=0.2)
parser.add_argument("--epsilon", type=float, default=1.0)


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
    model_config["max_alpha"]=args.max_alpha
    vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    vocab = load_vocab(vocab_file=vocab_file)
    if model_config['loss']=="wce":
        class_weights = token_weights(vocab_file)
        class_weights = torch.FloatTensor(class_weights)
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, loss_weights=class_weights, add_latent=add_latent)
    else: model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, add_latent=add_latent)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

class PropertyPrediction():
    def __init__(self, model, nr_vars, objective_type):
        self.model_predictor = model
        self.weight_electron_affinity = 1  # Adjust the weight for electron affinity
        self.weight_ionization_potential = 1  # Adjust the weight for ionization potential
        self.weight_z_distances = 5  # Adjust the weight for distance between GA chosen z and reencoded z
        self.penalty_value = -10  # Adjust the weight for penalty of validity
        self.results_custom = {}
        self.nr_vars = nr_vars
        self.eval_calls = 0
        self.objective_type = objective_type

    def evaluate(self, **params):
        # Assuming x is a 1D array containing the 32 numerical parameters

        # Inference: forward pass NN prediciton of properties and beam search decoding from latent
        #x = torch.from_numpy(np.array(list(params.values()))).to(device).to(torch.float32)
        self.eval_calls += 1
        print(params)
        _vector = [params[f'x{i}'] for i in range(self.nr_vars)]
        print(_vector)
        x = torch.from_numpy(np.array(_vector)).to(device).to(torch.float32)
        with torch.no_grad():
            predictions, _, _, _, y = self.model_predictor.inference(data=x, device=device, sample=False, log_var=None)
        # Validity check of the decoded molecule + penalize invalid molecules
        prediction_strings, validity = self._calc_validity(predictions)
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        prediction_strings_valid = [j for j, valid in zip(prediction_strings, validity) if valid]
        y_p_after_encoding_valid, z_p_after_encoding_valid, all_reconstructions_valid, _ = self._encode_and_predict_decode_molecules(predictions_valid)
        invalid_mask = (validity == 0)
        # Encode and predict the valid molecules
        expanded_y_p = np.array([y_p_after_encoding_valid.pop(0) if val == 1 else [np.nan,np.nan] for val in list(validity)])
        expanded_z_p = np.array([z_p_after_encoding_valid.pop(0) if val == 1 else [0] * 32 for val in list(validity)])
        #print(x, expanded_z_p)
        if self.objective_type=='EAmin':
            obj1 = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]
            obj2 = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1)
        if self.objective_type=='mimick_peak':
            obj1 = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2)
            obj2 = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.2)
        if self.objective_type=='mimick_best':
            obj1 = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2.64)
            obj2 = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.61)
        if self.objective_type=='max_gap':
            obj1 = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]
            obj2 = - self.weight_ionization_potential * expanded_y_p[~invalid_mask, 1]
        #latent_inconsistency = np.linalg.norm(x.detach().numpy()-expanded_z_p)
        #print(latent_inconsistency)

        if validity[0]:
            obj3=0
            aggr_obj = -(obj1[0]+obj2[0]+obj3)#-latent_inconsistency
        else:
            obj3 = self.penalty_value
            aggr_obj = obj3#-latent_inconsistency
        
        # results

        results_dict = {
            "objective":aggr_obj,
            "latents_BO": x,
            "latents_reencoded": expanded_z_p, 
            "predictions_BO": y,
            "predictions_reencoded": expanded_y_p,
            "string_decoded": prediction_strings, 
            "string_reconstructed": all_reconstructions_valid,
        }
        self.results_custom[str(self.eval_calls)] = results_dict
        print(results_dict)

        #Aggregate the objectives to do SOO with bayesian optimization
        return aggr_obj

    def _make_polymer_mol(self,poly_input):
        # If making the mol works, the string is considered valid
        try: 
            _ = (make_polymer_mol(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1]), poly_input.split("<")[1:])
            return 1
        # If not, it is considered invalid
        except: 
            return 0
    
    def _calc_validity(self, predictions):
        # Molecule validity check     
        # Return a boolean array indicating whether each solution is valid or not
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        mols_valid= []
        for _s in prediction_strings:
            poly_input = _s[:-1] # Last element is the _ char
            poly_input_nocan=None
            poly_label1 = np.nan
            poly_label2 = np.nan
            try: 
                poly_graph=poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan)
                mols_valid.append(1)
            except:
                poly_graph = None
                mols_valid.append(0)
        mols_valid = np.array(mols_valid) # List of lists
        return prediction_strings, mols_valid
    
    def _encode_and_predict_decode_molecules(self, predictions):
        # create data that can be encoded again
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        data_list = []
        print(prediction_strings)
        for i, s in enumerate(prediction_strings):
            poly_input = s[:-1] # Last element is the _ char
            poly_input_nocan=None
            poly_label1 = np.nan
            poly_label2 = np.nan
            g = poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan)
            if tokenization=="oldtok":
                target_tokens = tokenize_poly_input(poly_input=poly_input)
            elif tokenization=="RT_tokenized":
                target_tokens = tokenize_poly_input_RTlike(poly_input=poly_input)
            tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=target_tokens, vocab=vocab)
            g.tgt_token_ids = tgt_token_ids
            g.tgt_token_lens = tgt_lens
            g.to(device)
            data_list.append(g)
        data_loader = DataLoader(dataset=data_list, batch_size=64, shuffle=False)
        dict_data_loader = MP_Matrix_Creator(data_loader, device)

        #Encode and predict
        batches = list(range(len(dict_data_loader)))
        y_p = []
        z_p = []
        all_reconstructions = []
        with torch.no_grad():
            for i, batch in enumerate(batches):
                data = dict_data_loader[str(batch)][0]
                data.to(device)
                dest_is_origin_matrix = dict_data_loader[str(batch)][1]
                dest_is_origin_matrix.to(device)
                inc_edges_to_atom_matrix = dict_data_loader[str(batch)][2]
                inc_edges_to_atom_matrix.to(device)

                # Perform a single forward pass.
                reconstruction, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
                y_p.append(y.cpu().numpy())
                z_p.append(z.cpu().numpy())
                reconstruction_strings = [combine_tokens(tokenids_to_vocab(reconstruction[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(reconstruction))]
                all_reconstructions.extend(reconstruction_strings)
        #Return the predictions from the encoded latents
        y_p_flat = [sublist.tolist() for array_ in y_p for sublist in array_]
        z_p_flat = [sublist.tolist() for array_ in z_p for sublist in array_]
        self.modified_solution = z_p_flat

        return y_p_flat, z_p_flat, all_reconstructions, dict_data_loader

# Determine the boundaries for the latent dimensions from training dataset
dir_name = os.path.join(main_dir_path,'Checkpoints/', model_name)

with open(dir_name+'latent_space_'+dataset_type+'.npy', 'rb') as f:
    latent_space = np.load(f)
min_values = np.amin(latent_space, axis=0).tolist()
max_values = np.amax(latent_space, axis=0).tolist()

cutoff=-0.05

if not cutoff==0.0:
    transformed_min_values = []
    transformed_max_values = []
    for min_val, max_val in zip(min_values, max_values):
        #bounds are larger than in training set if cutoff value is negative
        # Calculate amount to cut off from each end (5%)
        cutoff_amount = cutoff * abs(max_val - min_val)
        # Adjust min and max values
        transformed_min = min_val + cutoff_amount
        transformed_max = max_val - cutoff_amount
        transformed_min_values.append(transformed_min)
        transformed_max_values.append(transformed_max)
    bounds = {'x{}'.format(i): (j, k) for i,(j,k) in enumerate(zip(transformed_min_values,transformed_max_values))}
elif cutoff==0: 
    bounds = {'x{}'.format(i): (j, k) for i,(j,k) in enumerate(zip(min_values,max_values))}



nr_vars = 32
objective_type='EAmin' # options: max_gap, EAmin, mimick_peak, mimick_best
prop_predictor = PropertyPrediction(model, nr_vars, objective_type)

# Initialize BayesianOptimization
optimizer = BayesianOptimization(f=prop_predictor.evaluate, pbounds=bounds)

# Perform optimization
# Perform optimization
#utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.01)
utility = UtilityFunction(kind="ucb")

# Define the time limit in seconds
stopping_criterion = "time" # time or iter
max_time = 600  # Set to 600 seconds, for example

# Run the optimizer with the callback
# Custom modification of the maximize function: If the max_time argument is specified, the 
optimizer.maximize(init_points=5, n_iter=100, acquisition_function=utility, max_time=max_time)

#optimizer.maximize(init_points=20, n_iter=500, acquisition_function=utility)
results = optimizer.res
results_custom = prop_predictor.results_custom

with open(dir_name+'optimization_results_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.pkl', 'wb') as f:
    pickle.dump(results, f)
with open(dir_name+'optimization_results_custom_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.pkl', 'wb') as f:
    pickle.dump(results_custom, f)



# Get the best parameters found
best_params = optimizer.max['params']
best_objective = optimizer.max['target']

print("Best Parameters:", best_params)
print("Best Objective Value:", best_objective)

with open(dir_name+'optimization_results_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.pkl', 'rb') as f:
    results = pickle.load(f)
with open(dir_name+'optimization_results_custom_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.pkl', 'rb') as f:
    results_custom = pickle.load(f)

with open(dir_name+'optimization_results_custom_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.txt', 'w') as fl:
     print(results_custom, file=fl)
#print(results_custom)
# Calculate distances between the BO and reencoded latents
Latents_BO = []
Latents_RE = []
latent_inconsistencies = []
pred_BO = []
pred_RE = []
decoded_mols= []
rec_mols=[]

for eval, res in results_custom.items():
    eval_int= int(eval)
    L_bo= res["latents_BO"].detach().numpy()
    L_re=res["latents_reencoded"][0]
    print(L_bo)
    print(L_re)
    latent_inconsistency = np.linalg.norm(L_bo-L_re)
    latent_inconsistencies.append(latent_inconsistency)
    Latents_BO.append(L_bo)
    Latents_RE.append(L_re)
    pred_BO.append(res["predictions_BO"][0])
    pred_RE.append(res["predictions_reencoded"][0])
    decoded_mols.append(res["string_decoded"][0])
    if not len(res["string_reconstructed"])==0:
        rec_mols.append(res["string_reconstructed"][0])
    else: rec_mols.append("Invalid decoded molecule")


def distance_matrix(arrays):
    num_arrays = len(arrays)
    dist_matrix = np.zeros((num_arrays, num_arrays))

    for i in range(num_arrays):
        for j in range(num_arrays):
            dist_matrix[i, j] = np.linalg.norm(arrays[i] - arrays[j])
    # Flatten upper triangular part of the matrix (excluding diagonal)
    flattened_upper_triangular = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]

    # Calculate mean and standard deviation
    mean_distance = np.mean(flattened_upper_triangular)
    std_distance = np.std(flattened_upper_triangular)

    return dist_matrix, mean_distance, std_distance

dist_matrix_zBO, mBO, sBO=distance_matrix(Latents_BO)
dist_matrix_zRE, mRE, sRE=distance_matrix(Latents_RE)
print(mBO, sBO)
print(mRE, sRE)
print(np.mean(latent_inconsistencies), np.std(latent_inconsistencies))

import matplotlib.pyplot as plt

iterations = range(len(pred_BO))
EA_bo= [arr[0] for arr in pred_BO]
IP_bo = [arr[1] for arr in pred_BO]
EA_re= [arr[0] for arr in pred_RE]
IP_re = [arr[1] for arr in pred_RE]

# Create plot
plt.figure(0)

plt.plot(iterations, EA_bo, label='EA (BO)')
plt.plot(iterations, IP_bo, label='IP (BO)')
plt.plot(iterations, EA_re, label='EA (RE)')
plt.plot(iterations, IP_re, label='IP (RE)')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.savefig(dir_name+'BO_objectives_'+str(cutoff)+'_'+str(stopping_criterion)+'.png',  dpi=300)
plt.close()


# Plot the training data pca and optimization points
    
# Dimensionality reduction
#PCA
# 1. Load trained latent space
dataset_type = "train"
with open(dir_name+'latent_space_'+dataset_type+'.npy', 'rb') as f:
    latent_space_train = np.load(f)
with open(dir_name+'y1_all_'+dataset_type+'.npy', 'rb') as f:
    y1_all_train = np.load(f)
#2. load fitted pca
dim_red_type="pca"
with open(dir_name+dim_red_type+'_fitted_train', 'rb') as f:
    reducer = pickle.load(f)

# 3. Transform train LS   
z_embedded_train = reducer.transform(latent_space_train)
# 4. Transform points of Optimization
latents_BO_np = np.stack(Latents_BO)
z_embedded_BO = reducer.transform(latents_BO_np)

latents_RE_np = np.stack(Latents_RE)
z_embedded_RE = reducer.transform(latents_RE_np)
plt.figure(1)


plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y1_all_train, cmap='viridis')
clb = plt.colorbar()
clb.ax.set_title('Electron affinity')
plt.scatter(z_embedded_BO[:, 0], z_embedded_BO[:, 1], s=2, c='black')
plt.scatter(z_embedded_RE[:, 0], z_embedded_RE[:, 1], s=2, c='red')
plt.savefig(dir_name+'BO_projected_to_pca_'+str(cutoff)+'_'+str(stopping_criterion)+'.png',  dpi=300)
plt.close()
#pca = PCA(n_components=2)


### Do the same but only for improved points
def indices_of_improvement(values):
    indices_of_increases = []

    # Initialize the highest value and its index
    highest_value = values[0]
    highest_index = 0

    # Iterate through the values
    for i, value in enumerate(values):
        # If the current value is greater than the previous highest value
        if value > highest_value:
            highest_value = value  # Update the highest value
            highest_index = i      # Update the index of the highest value
            indices_of_increases.append(i)  # Save the index of increase

    return indices_of_increases

def top_n_molecule_indices(objective_values, n_idx=10):
    # Get the indices of 20 molecules with the highest objective values
    # Pair each value with its index
    # Filter out NaN values and keep track of original indices
    filtered_indexed_values = [(index, value) for index, value in enumerate(objective_values) if not math.isnan(value)]
    # Sort the indexed values by the value in descending order and take n_idx best ones
    sorted_filtered_indexed_values = sorted(filtered_indexed_values, key=lambda x: x[1], reverse=True)
    top_idxs = [index for index, value in sorted_filtered_indexed_values[:n_idx]]

    return top_idxs

# Extract data for the curves
if objective_type=='mimick_peak':
    objective_values = [-(np.abs(arr[0]+2)+np.abs(arr[1]-1.2)) for arr in pred_RE]
elif objective_type=='mimick_best':
    objective_values = [-((np.abs(arr[0]+2.64)+np.abs(arr[1]-1.61))) for arr in pred_RE]
elif objective_type=='EAmin': 
    objective_values = [-(arr[0]+np.abs(arr[1]-1)) for arr in pred_RE]
elif objective_type =='max_gap':
    objective_values = [-(arr[0]-arr[1]) for arr in pred_RE]

indices_of_increases = indices_of_improvement(objective_values)

EA_bo_imp = [EA_bo[i] for i in indices_of_increases]
IP_bo_imp = [IP_bo[i] for i in indices_of_increases]
EA_re_imp = [EA_re[i] for i in indices_of_increases]
IP_re_imp = [IP_re[i] for i in indices_of_increases]
best_z_re = [Latents_RE[i] for i in indices_of_increases]
best_mols = {i+1: decoded_mols[i] for i in indices_of_increases}
best_props = {i+1: [EA_re[i], EA_bo[i], IP_re[i], IP_bo[i]] for i in indices_of_increases}
best_mols_rec = {i+1: rec_mols[i] for i in indices_of_increases}
with open(dir_name+'best_mols_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.txt', 'w') as fl:
    print(best_mols, file=fl)
    print(best_props, file=fl)
with open(dir_name+'best_recon_mols_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.txt', 'w') as fl:
    print(best_mols_rec, file=fl)

top_20_indices = top_n_molecule_indices(objective_values, n_idx=20)
best_mols_t20 = {i+1: decoded_mols[i] for i in top_20_indices}
best_props_t20 = {i+1: [EA_re[i], EA_bo[i], IP_re[i], IP_bo[i]] for i in top_20_indices}
with open(dir_name+'top20_mols_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.txt', 'w') as fl:
    print(best_mols_t20, file=fl)
    print(best_props_t20, file=fl)


print(objective_values)
print(indices_of_improvement)
latents_BO_np_imp = np.stack([Latents_BO[i] for i in indices_of_increases])
z_embedded_BO_imp = reducer.transform(latents_BO_np_imp)

latents_RE_np_imp = np.stack([Latents_RE[i] for i in indices_of_increases])
z_embedded_RE_imp = reducer.transform(latents_RE_np_imp)

plt.figure(2)


plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y1_all_train, cmap='viridis', alpha=0.2)
clb = plt.colorbar()
clb.ax.set_title('Electron affinity')
#plt.scatter(z_embedded_BO[:, 0], z_embedded_BO[:, 1], s=1, c='black', marker="1")
#plt.scatter(z_embedded_RE[:, 0], z_embedded_RE[:, 1], s=1, c='red',marker="2")
# Real latent space (reencoded)
for i, (x, y) in enumerate(z_embedded_RE_imp):
    it=indices_of_increases[i]
    plt.scatter(x, y, color='red', s=3,  marker="2")  # Plot points
    plt.text(x, y+0.2, f'{i+1}({it+1})', fontsize=6, color="red", ha='center', va='center')  # Annotate with labels

# Connect points with arrows
for i in range(len(z_embedded_RE_imp) - 1):
    x_start, y_start = z_embedded_RE_imp[i]
    x_end, y_end = z_embedded_RE_imp[i + 1]
    plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, 
              shape='full', lw=0.05, length_includes_head=True, head_width=0.1, color='red')


for i, (x, y) in enumerate(z_embedded_BO_imp):
    it=indices_of_increases[i]
    plt.scatter(x, y, color='black', s=2,  marker="1")  # Plot points
    plt.text(x, y+0.2, f'{i+1}', fontsize=6, color="black", ha='center', va='center')  # Annotate with labels

# Connect points with arrows
for i in range(len(z_embedded_BO_imp) - 1):
    x_start, y_start = z_embedded_BO_imp[i]
    x_end, y_end = z_embedded_BO_imp[i + 1]
    plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, 
              shape='full', lw=0.05, length_includes_head=True, head_width=0.1, color='black')

# Set plot labels and title
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.title('Optimization in latent space')

plt.savefig(dir_name+'BO_imp_projected_to_pca_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.png',  dpi=300)

plt.figure(3)


plt.scatter(z_embedded_train[:, 0], z_embedded_train[:, 1], s=1, c=y1_all_train, cmap='viridis', alpha=0.2)
clb = plt.colorbar()
clb.ax.set_title('Electron affinity')
#plt.scatter(z_embedded_BO[:, 0], z_embedded_BO[:, 1], s=1, c='black', marker="1")
#plt.scatter(z_embedded_RE[:, 0], z_embedded_RE[:, 1], s=1, c='red',marker="2")
# Real latent space (reencoded)
for i, (x, y) in enumerate(z_embedded_RE_imp):
    it=indices_of_increases[i]
    plt.scatter(x, y, color='red', s=3,  marker="2")  # Plot points
    plt.text(x, y+0.2, f'{i+1}({it+1})', fontsize=6, color="red", ha='center', va='center')  # Annotate with labels

# Connect points with arrows
for i in range(len(z_embedded_RE_imp) - 1):
    x_start, y_start = z_embedded_RE_imp[i]
    x_end, y_end = z_embedded_RE_imp[i + 1]
    plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, 
              shape='full', lw=0.05, length_includes_head=True, head_width=0.1, color='red')

# Set plot labels and title
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.title('Optimization in latent space')

plt.savefig(dir_name+'BO_imp_projected_to_pca_onlyred_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.png',  dpi=300)



""" Sample around seed molecule - seed being the optimal solution found by optimizer """
# Sample around the optimal molecule and predict the property values

all_prediction_strings=[]
all_reconstruction_strings=[]
seed_z = best_z_re[-1] # last reencoded best molecule
# extract the predictions of best molecule together with the latents of best molecule(by encoding? or BO one)
seed_z = torch.from_numpy(np.array(seed_z)).unsqueeze(0).repeat(64,1).to(device).to(torch.float32)
print(seed_z)
with torch.no_grad():
    predictions, _, _, _, y_seed = model.inference(data=seed_z, device=device, sample=False, log_var=None)
    seed_string = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
sampled_z = []
all_y_p = []
#model.eval()

with torch.no_grad():
    # Define the mean and standard deviation of the Gaussian noise
    for r in range(8):
        mean = 0
        std = args.epsilon
        
        # Create a tensor of the same size as the original tensor with random noise
        noise = torch.tensor(np.random.normal(mean, std, size=seed_z.size()), dtype=torch.float, device=device)

        # Add the noise to the original tensor
        seed_z_noise = seed_z + noise
        sampled_z.append(seed_z_noise.cpu().numpy())
        predictions, _, _, _, y = model.inference(data=seed_z_noise, device=device, sample=False, log_var=None)
        prediction_strings, validity = prop_predictor._calc_validity(predictions)
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        prediction_strings_valid = [j for j, valid in zip(prediction_strings, validity) if valid]
        y_p_after_encoding_valid, z_p_after_encoding_valid, reconstructions_valid, _ = prop_predictor._encode_and_predict_decode_molecules(predictions_valid)
        all_prediction_strings.extend(prediction_strings_valid)
        all_reconstruction_strings.extend(reconstructions_valid)
        all_y_p.extend(y_p_after_encoding_valid)


print(f'Saving generated strings')
i=0
with open(dir_name+'results_around_BO_seed_'+str(cutoff)+'_'+str(objective_type)+'_'+str(stopping_criterion)+'.txt', 'w') as f:
    f.write("Seed string decoded: " + seed_string[0] + "\n")
    f.write("Prediction: "+ str(y_seed[0]))
    f.write("The results, sampling around the best population are the following\n")
    for l1,l2 in zip(all_reconstruction_strings,all_prediction_strings):
        if l1 == l2:
            f.write("decoded molecule from GA selection is encoded and decoded to the same molecule\n")
            f.write(l1 + "\n")
            f.write("The predicted properties from z(reencoded) are: " + str(all_y_p[i]) + "\n")
        else:
            f.write("GA molecule: ")
            f.write(l2 + "\n")
            f.write("Encoded and decoded GA molecule: ")
            f.write(l1 + "\n")
            f.write("The predicted properties from z(reencoded) are: " + str(all_y_p[i]) + "\n")
        i+=1

""" Check the molecules around the optimized seed """

def poly_smiles_to_molecule(poly_input):
    '''
    Turns adjusted polymer smiles string into PyG data objects
    '''

    # Turn into RDKIT mol object
    mols = make_monomer_mols(poly_input.split("|")[0], 0, 0,  # smiles
                            fragment_weights=poly_input.split("|")[1:-1])
    
    return mols

def valid_scores(smiles):
    return np.array(list(map(make_polymer_mol, smiles)), dtype=np.float32)

dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')
data_augment ="old"
vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
vocab = load_vocab(vocab_file=vocab_file)

all_predictions=all_prediction_strings.copy()
sm_can = SmilesEnumCanon()
all_predictions_can = list(map(sm_can.canonicalize, all_predictions))
prediction_validityA= []
prediction_validityB =[]
data_dir = os.path.join(main_dir_path,'data/')


"""
if augment=="augmented":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-poly_chemprop_v2.csv')
elif augment=="augmented_canonical":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-canonical-poly_chemprop.csv')
elif augment=="augmented_enum":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-enumerated2_poly_chemprop.csv')
all_polymers_data= []
all_train_polymers = []

for batch, graphs in enumerate(dict_train_loader):
    data = dict_train_loader[str(batch)][0]
    train_polymers_batch = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization).split('_')[0] for sample in range(len(data))]
    all_train_polymers.extend(train_polymers_batch)
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    all_polymers_data.append(poly_input)

 
all_predictions_can = list(map(sm_can.canonicalize, all_predictions))
all_train_can = list(map(sm_can.canonicalize, all_train_polymers))
all_pols_data_can = list(map(sm_can.canonicalize, all_polymers_data))
monomers= [s.split("|")[0].split(".") for s in all_train_polymers]
monomers_all=[mon for sub_list in monomers for mon in sub_list]
all_mons_can = []
for m in monomers_all:
    m_can = sm_can.canonicalize(m, monomer_only=True, stoich_con_info=False)
    modified_string = re.sub(r'\*\:\d+', '*', m_can)
    all_mons_can.append(modified_string)
all_mons_can = list(set(all_mons_can))
print(len(all_mons_can), all_mons_can[1:3])




with open(data_dir+'all_train_pols_can'+'.pkl', 'wb') as f:
    pickle.dump(all_train_can, f)
with open(data_dir+'all_pols_data_can'+'.pkl', 'wb') as f:
    pickle.dump(all_pols_data_can, f)
with open(data_dir+'all_mons_train_can'+'.pkl', 'wb') as f:
    pickle.dump(all_mons_can, f) """

with open(data_dir+'all_train_pols_can'+'.pkl', 'rb') as f:
    all_train_can = pickle.load(f)
with open(data_dir+'all_pols_data_can'+'.pkl', 'rb') as f:
    all_pols_data_can= pickle.load(f)
with open(data_dir+'all_mons_train_can'+'.pkl', 'rb') as f:
    all_mons_can = pickle.load(f)


# C
prediction_mols = list(map(poly_smiles_to_molecule, all_predictions))
for mon in prediction_mols: 
    try: prediction_validityA.append(mon[0] is not None)
    except: prediction_validityA.append(False)
    try: prediction_validityB.append(mon[1] is not None)
    except: prediction_validityB.append(False)


#predBvalid = []
#for mon in prediction_mols:
#    try: 
#        predBvalid.append(mon[1] is not None)
#    except: 
#        predBvalid.append(False)

#prediction_validityB.append(predBvalid)
#reconstructed_SmilesA = list(map(Chem.MolToSmiles, [mon[0] for mon in prediction_mols]))
#reconstructed_SmilesB = list(map(Chem.MolToSmiles, [mon[1] for mon in prediction_validity]))


# Evaluation of validation set reconstruction accuracy (inference)
monomer_smiles_predicted = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
monomer_comb_predicted = [poly_smiles.split("|")[0] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
monomer_comb_train = [poly_smiles.split("|")[0] for poly_smiles in all_train_can if poly_smiles]

monA_pred = [mon[0] for mon in monomer_smiles_predicted]
monB_pred = [mon[1] for mon in monomer_smiles_predicted]
monA_pred_gen = []
monB_pred_gen = []
for m_c in monomer_smiles_predicted:
    ma = m_c[0]
    mb = m_c[1]
    ma_can = sm_can.canonicalize(ma, monomer_only=True, stoich_con_info=False)

    monA_pred_gen.append(re.sub(r'\*\:\d+', '*', ma_can))
    mb_can = sm_can.canonicalize(mb, monomer_only=True, stoich_con_info=False)
    monB_pred_gen.append(re.sub(r'\*\:\d+', '*', mb_can))

monomer_weights_predicted = [poly_smiles.split("|")[1:-1] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
monomer_con_predicted = [poly_smiles.split("|")[-1].split("_")[0] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']


#prediction_validityA= [num for elem in prediction_validityA for num in elem]
#prediction_validityB = [num for elem in prediction_validityB for num in elem]
validityA = sum(prediction_validityA)/len(prediction_validityA)
validityB = sum(prediction_validityB)/len(prediction_validityB)

# Novelty metrics
novel = 0
novel_pols=[]
for pol in monomer_comb_predicted:
    if not pol in monomer_comb_train:
        novel+=1
        novel_pols.append(pol)
novelty_mon_comb = novel/len(monomer_comb_predicted)
novel = 0
novel_pols=[]
for pol in all_predictions_can:
    if not pol in all_train_can:
        novel+=1
        novel_pols.append(pol)
novelty = novel/len(all_predictions_can)
novel = 0
for pol in all_predictions_can:
    if not pol in all_pols_data_can:
        novel+=1
novelty_full_dataset = novel/len(all_predictions_can)
novelA = 0
novelAs = []
for monA in monA_pred_gen:
    if not monA in all_mons_can:
        novelA+=1
        novelAs.append(monA)
print(novelAs, len(list(set(novelAs))))
novelty_A = novelA/len(monA_pred_gen)
novelB = 0
novelBs = []
for monB in monB_pred_gen:
    if not monB in all_mons_can:
        novelB+=1
        novelBs.append(monB)
print(novelBs, len(list(set(novelBs))))

novelty_B = novelB/len(monB_pred_gen)

diversity = len(list(set(all_predictions_can)))/len(all_predictions_can)
diversity_novel = len(list(set(novel_pols)))/len(novel_pols)

classes_stoich = [['0.5','0.5'],['0.25','0.75'],['0.75','0.25']]
#if data_augment=='new':
#    classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
#else:
classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
whole_valid = len(monomer_smiles_predicted)
validity = whole_valid/len(all_predictions)
with open(dir_name+'novelty_BO_seed_'+str(objective_type)+'_'+str(stopping_criterion)+'.txt', 'w') as f:
    f.write("Gen Mon A validity: %.4f %% Gen Mon B validity: %.4f %% "% (100*validityA, 100*validityB,))
    f.write("Gen validity: %.4f %% "% (100*validity,))
    f.write("Novelty: %.4f %% "% (100*novelty,))
    f.write("Novelty (mon_comb): %.4f %% "% (100*novelty_mon_comb,))
    f.write("Novelty MonA full dataset: %.4f %% "% (100*novelty_A,))
    f.write("Novelty MonB full dataset: %.4f %% "% (100*novelty_B,))
    f.write("Novelty in full dataset: %.4f %% "% (100*novelty_full_dataset,))
    f.write("Diversity: %.4f %% "% (100*diversity,))
    f.write("Diversity (novel polymers): %.4f %% "% (100*diversity_novel,))


""" Plot the kde of the properties of training data and sampled data """
from sklearn.neighbors import KernelDensity

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
yp1_all_seed = [yp[0] for yp in all_y_p]
yp2_all_seed = [yp[1] for yp in all_y_p]
# Do a KDE to check if the distributions of properties are similar (predicted vs. real lables)
""" y1 """
plt.figure(4)
real_distribution = np.array([r for r in y1_all if not np.isnan(r)])
augmented_distribution = np.array([p for p in yp1_all])
seed_distribution = np.array([s for s in yp1_all_seed])


# Reshape the data
real_distribution = real_distribution.reshape(-1, 1)
augmented_distribution = augmented_distribution.reshape(-1, 1)
seed_distribution = seed_distribution.reshape(-1, 1)

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
kde_sampled_seed.fit(seed_distribution)

# Create a range of values for the x-axis
x_values = np.linspace(min(np.min(real_distribution), np.min(augmented_distribution), np.min(seed_distribution)), max(np.max(real_distribution), np.max(augmented_distribution), np.max(seed_distribution)), 1000)
# Evaluate the KDE on the range of values
real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
seed_density = np.exp(kde_sampled_seed.score_samples(x_values.reshape(-1, 1)))

# Plotting
plt.plot(x_values, real_density, label='Real Data')
plt.plot(x_values, augmented_density, label='Augmented Data')
plt.plot(x_values, seed_density, label='Sampled around optimal molecule')

plt.xlabel('EA (eV)')
plt.ylabel('Density')
plt.title('Kernel Density Estimation (Electron affinity)')
plt.legend()
plt.show()
plt.savefig(dir_name+'KDEy1_BO_seed'+'_'+str(stopping_criterion)+'.png')

""" y2 """
plt.figure(5)
real_distribution = np.array([r for r in y2_all if not np.isnan(r)])
augmented_distribution = np.array([p for p in yp2_all])
seed_distribution = np.array([s for s in yp2_all_seed])


# Reshape the data
real_distribution = real_distribution.reshape(-1, 1)
augmented_distribution = augmented_distribution.reshape(-1, 1)
seed_distribution = seed_distribution.reshape(-1, 1)

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
kde_sampled_seed.fit(seed_distribution)

# Create a range of values for the x-axis
x_values = np.linspace(min(np.min(real_distribution), np.min(augmented_distribution), np.min(seed_distribution)), max(np.max(real_distribution), np.max(augmented_distribution), np.max(seed_distribution)), 1000)
# Evaluate the KDE on the range of values
real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
seed_density = np.exp(kde_sampled_seed.score_samples(x_values.reshape(-1, 1)))

# Plotting
plt.plot(x_values, real_density, label='Real Data')
plt.plot(x_values, augmented_density, label='Augmented Data')
plt.plot(x_values, seed_density, label='Sampled around optimal molecule')
plt.xlabel('IP (eV)')
plt.ylabel('Density')
plt.title('Kernel Density Estimation (Ionization potential)')
plt.legend()
plt.show()
plt.savefig(dir_name+'KDEy2_BO_seed'+'_'+str(stopping_criterion)+'.png')