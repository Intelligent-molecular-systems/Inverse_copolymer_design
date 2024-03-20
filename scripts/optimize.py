import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.core.termination import Termination
import argparse
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from scipy.spatial import distance
import pickle

import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *
from data_processing.Function_Featurization_Own import poly_smiles_to_graph
from data_processing.rdkit_poly import make_polymer_mol


# setting device on GPU if available, else CPU
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
    model.eval()

dir_name = os.path.join(main_dir_path,'Checkpoints/', model_name)


def MP_Matrix_Creator(loader):
    '''
    Here we create the two matrices needed later for the message passinng part of the graph neural network. 
    They are in essence different forms of the adjacency matrix of the graph. They are created on a batch thus the batches cannot be shuffled
    The graph and both matrices are saved per batch in a dictionary
    '''
    dict_graphs_w_matrix = {}
    for batch, graph in enumerate(loader):
        # get attributes of graphs in batch
        nodes = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        atom_weights = graph.W_atoms
        bond_weights = graph.W_bonds
        num_bonds = edge_index[0].shape[0]

        '''
        Create edge update message passing matrix
        '''
        dest_is_origin_matrix = torch.zeros(
            size=(num_bonds, num_bonds)).to(device)
        # for sparse matrix
        I = torch.empty(2, 0, dtype=torch.long).to(device)
        V = torch.empty(0).to(device)

        for i in range(num_bonds):
            # find edges that are going to the originating atom (neigbouring edges)
            incoming_edges_idx = (
                edge_index[1] == edge_index[0, i]).nonzero().flatten()
            # check whether those edges originate from our bonds destination atom, if so ignore that bond
            idx_from_dest_atom = (
                edge_index[0, incoming_edges_idx] == edge_index[1, i])
            incoming_edges_idx = incoming_edges_idx[idx_from_dest_atom != True]
            # find the features and assoociated weights of those neigbouring edges
            weights_inc_edges = bond_weights[incoming_edges_idx]
            # create matrix
            dest_is_origin_matrix[i, incoming_edges_idx] = weights_inc_edges

            # For Sparse Version
            edge = torch.tensor([i])
            # create indices
            i1 = edge.repeat_interleave(len(incoming_edges_idx)).to(device)
            i2 = incoming_edges_idx.clone().to(device)
            i = torch.stack((i1, i2), dim=0)
            # find assocociated values
            v = weights_inc_edges

            # append to larger arrays
            I = torch.cat((I, i), dim=1)
            V = torch.cat((V, v))

        # create a COO sparse version of edge message passing matrix
        dest_is_origin_sparse = torch.sparse_coo_tensor(
            I, V, [num_bonds, num_bonds])
        '''
        Create node update message passing matrix
        '''
        inc_edges_to_atom_matrix = torch.zeros(
            size=(nodes.shape[0], edge_index.shape[1])).to(device)

        I = torch.empty(2, 0, dtype=torch.long).to(device)
        V = torch.empty(0).to(device)
        for i in range(nodes.shape[0]):
            # find index of edges that are incoming to specific atom
            inc_edges_idx = (edge_index[1] == i).nonzero().flatten()
            weights_inc_edges = bond_weights[inc_edges_idx]
            inc_edges_to_atom_matrix[i, inc_edges_idx] = weights_inc_edges

            # for sparse version
            node = torch.tensor([i]).to(device)
            i1 = node.repeat_interleave(len(inc_edges_idx))
            i2 = inc_edges_idx.clone()
            i = torch.stack((i1, i2), dim=0)
            v = weights_inc_edges

            I = torch.cat((I, i), dim=1).to(device)
            V = torch.cat((V, v)).to(device)

        # create a COO sparse version of node message passing matrix
        inc_edges_to_atom_sparse = torch.sparse_coo_tensor(
            I, V, [nodes.shape[0], edge_index.shape[1]])

        if batch % 10 == 0:
            print(f"[{batch} / {len(loader)}]")

        '''
        Store in Dictionary
        '''
        dict_graphs_w_matrix[str(batch)] = [
            graph, dest_is_origin_sparse, inc_edges_to_atom_sparse]

    return dict_graphs_w_matrix

class Property_optimization_problem(Problem):
    def __init__(self, model, x_min, x_max):
        super().__init__(n_var=len(x_min), n_obj=2, n_constr=0, xl=x_min, xu=x_max)
        self.model_predictor = model
        self.weight_electron_affinity = 1  # Adjust the weight for electron affinity
        self.weight_ionization_potential = 1  # Adjust the weight for ionization potential
        self.weight_z_distances = 5  # Adjust the weight for distance between GA chosen z and reencoded z
        self.penalty_value = 100  # Adjust the weight for penalty of validity
        self.modified_solution = None # Initialize the class variable that later stores the recalculated latents
        self.modified_solution_history = []  # Initialize list to store modified solutions


    def _evaluate(self, x, out, *args, **kwargs):
        # Assuming x is a 1D array containing the 32 numerical parameters

        # Inference: forward pass NN prediciton of properties and beam search decoding from latent
        x = torch.from_numpy(x).to(device).to(torch.float32) 
        with torch.no_grad():
            predictions, _, _, _, y = self.model_predictor.inference(data=x, device=device, sample=False, log_var=None)
        # Validity check of the decoded molecule + penalize invalid molecules
        validity = self._calc_validity(predictions)
        invalid_mask = (validity == 0)
        # Encode and predict the valid molecules
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        y_p_after_encoding_valid, z_p_after_encoding_valid=self._encode_and_predict_molecules(predictions_valid)
        expanded_y_p = np.array([y_p_after_encoding_valid.pop(0) if val == 1 else [np.nan,np.nan] for val in list(validity)])
        expanded_z_p = np.array([z_p_after_encoding_valid.pop(0) if val == 1 else [0] * 32 for val in list(validity)])
        #print(x, expanded_z_p)
        #dst = np.array([np.linalg.norm(a - b) for a,b in zip(x.cpu(), expanded_z_p)])
        #print(dst)
        # Use the encoded and predicted properties as evaluation for GA (more realistic property prediction)
        #out["F"] = np.zeros((x.shape[0], 3)) # for three objectives (additionally minimze the distance of z vectors)
        out["F"] = np.zeros((x.shape[0], 2))
        out["F"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]  # Minimize the first property (electron affinity)
        #out["F"][~invalid_mask, 0] = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2) # Bring the first property (electron affinity) close to -2
        out["F"][~invalid_mask, 1] = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1)  # Bring the second property (ionization potential) as close to 1 as possible
        #out["F"][~invalid_mask, 2] = self.weight_z_distances*dst[~invalid_mask] # Difference between the 
        out["F"][invalid_mask] = self.penalty_value  # Assign large penalty to invalid molecules
        out["X_mod"] = expanded_z_p


    def _calc_pareto_front(self, n_pareto_points=100):
        # Custom method to calculate Pareto front for visualization or comparison
        pass

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
        return mols_valid

    def _make_polymer_mol(self,poly_input):
        # If making the mol works, the string is considered valid
        try: 
            _ = (make_polymer_mol(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1]), poly_input.split("<")[1:])
            return 1
        # If not, it is considered invalid
        except: 
            return 0
    
    def _encode_and_predict_molecules(self, predictions):
        # create data that can be encoded again
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        data_list = []
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
        dict_data_loader = MP_Matrix_Creator(data_loader)

        #Encode and predict
        batches = list(range(len(dict_data_loader)))
        y_p = []
        z_p = []
        with torch.no_grad():
            for i, batch in enumerate(batches):
                data = dict_data_loader[str(batch)][0]
                data.to(device)
                dest_is_origin_matrix = dict_data_loader[str(batch)][1]
                dest_is_origin_matrix.to(device)
                inc_edges_to_atom_matrix = dict_data_loader[str(batch)][2]
                inc_edges_to_atom_matrix.to(device)

                # Perform a single forward pass.
                _, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
                y_p.append(y.cpu().numpy())
                z_p.append(z.cpu().numpy())
        #Return the predictions from the encoded latents
        y_p_flat = [sublist.tolist() for array_ in y_p for sublist in array_]
        z_p_flat = [sublist.tolist() for array_ in z_p for sublist in array_]
        self.modified_solution = z_p_flat

        return y_p_flat, z_p_flat


# Define a convergence termination class
class ConvergenceTermination(Termination):
    def __init__(self, conv_threshold, conv_generations, n_max_gen):
        super().__init__()
        self.conv_threshold = conv_threshold
        self.conv_generations = conv_generations
        self.n_max_gen = n_max_gen
        self.best_fit_values = []
        self.conv_counter = 0
        self.converged_solution_X = None
        self.converged_solution_F = None

    # check for the convergence criterion
    def _do_continue(self, algorithm):
        return self.perc < 1.0
    
    # check the convergence progress
    def _update(self, algorithm):
        best_fit = algorithm.pop.get("F").min()
        self.best_fit_values.append(best_fit)
        
        if algorithm.n_gen >= self.n_max_gen:
            return 1.0

        if len(self.best_fit_values) > self.conv_generations:
            conv_rate = abs(self.best_fit_values[-1] - self.best_fit_values[-self.conv_generations]) / self.conv_generations
            if conv_rate < self.conv_threshold:
                self.conv_counter += 1
                if self.conv_counter >= 5:
                    # store the termination object and use it to print the converged solution and the objective value later
                    self.converged_solution_X = algorithm.pop[np.argmin(algorithm.pop.get("F"))].get("X")
                    self.converged_solution_F = algorithm.pop.get("F")
                    print(f"Algorithm has converged after {algorithm.n_gen} generations.")
                    return 1.0
            else:
                self.conv_counter = 0
        return algorithm.n_gen / self.n_max_gen


# Define the callback function to store the generation number along with the non-dominated solutions found at each generation
def generation_callback(algorithm):
    #generation_number = algorithm.n_gen
    #non_dominated_solutions = algorithm.pop.get("F")
    # Store (X, X_mod, F) tuples of all solutions
    X = algorithm.pop.get("X")
    F = algorithm.pop.get("F")
    X_mod = algorithm.pop.get("X_mod")
    F_combined = np.sum(F, axis=1)
    #best_idx = np.argmin(F_combined)
    best_idx = np.argsort(F_combined)[:5] # Get the 5 best solutions per population
    all_solutions.extend([(x, x_mod, f) for x, x_mod, f in zip(X, X_mod, F)])

    # Find the index of the best solution in the current population
    
    # Store the (X, F) tuple of the best solution
    best_solutions.append((X[best_idx], X_mod[best_idx], F[best_idx]))


# Determine the boundaries for the latent dimensions from training dataset
with open(dir_name+'latent_space_'+dataset_type+'.npy', 'rb') as f:
    latent_space = np.load(f)
min_values = np.amin(latent_space, axis=0).tolist()
max_values = np.amax(latent_space, axis=0).tolist()

# Initialize the problem
problem = Property_optimization_problem(model, min_values, max_values)

# Termination criterium
termination = ConvergenceTermination(conv_threshold=0.1, conv_generations=5, n_max_gen=50)

# Initialize the NSGA2 algorithm
# Set up the NSGA-II algorithm
# Define NSGA2 algorithm parameters
pop_size = 100
sampling = LatinHypercubeSampling()
crossover = SimulatedBinaryCrossover(prob=0.9, eta=15)
mutation = PolynomialMutation(prob=1.0 / problem.n_var, eta=20)

# Initialize the NSGA2 algorithm
algorithm = NSGA2(pop_size=pop_size,
                  sampling=sampling,
                  crossover=crossover,
                  mutation=mutation,
                  eliminate_duplicates=True)

# Optimize the problem
all_solutions = []
best_solutions = []
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    callback=generation_callback,
    verbose=True,
)

# Access the results
best_solution = res.X
#best_mod_solution = res.X_mod
best_fitness = res.F

with open(dir_name+'res_optimization', 'wb') as f:
    pickle.dump(res, f)
with open(dir_name+'best_solutions', 'wb') as f:
    pickle.dump(best_solutions, f)
with open(dir_name+'all_solutions', 'wb') as f:
    pickle.dump(all_solutions, f)

#convergence = res.algorithm.termination
with open(dir_name+'res_optimization', 'rb') as f:
    res = pickle.load(f)
with open(dir_name+'best_solutions', 'rb') as f:
    best_solutions = pickle.load(f)
# decode the best solution
prediction_strings= []
predicted_y = []
with torch.no_grad():
    for b in best_solutions:
        bi = torch.from_numpy(b[0]).to(device).to(torch.float32)
        predictions, _, _, _, y = model.inference(data=bi, device=device, sample=False, log_var=None)
        prediction_s = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        prediction_strings.extend(prediction_s)
        predicted_y.append(y)

print(prediction_strings)
print(predicted_y)
with open(dir_name+'prediction_strings', 'wb') as f:
    pickle.dump(prediction_strings, f)
with open(dir_name+'predicted_y', 'wb') as f:
    pickle.dump(predicted_y, f)


# 1. convert the molecular strings to graphs again
# TODO: non valid strings need to be captured somehow
data_list = []
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

# Save the graphs in Dataloader and convert 
results_loader = DataLoader(dataset=data_list, batch_size=64, shuffle=False)
dict_results_loader = MP_Matrix_Creator(results_loader)
torch.save(dict_results_loader, dir_name+'/dict_results_loader.pt')

# 2. Encode + predict (forward pass) in eval mode (z=mean) + do inference decoding
#  Run over all epoch

# Start from here if optimization already run before

with open(dir_name+'res_optimization', 'rb') as f:
    res = pickle.load(f)
with open(dir_name+'best_solutions', 'rb') as f:
    best_solutions = pickle.load(f)
with open(dir_name+'prediction_strings', 'rb') as f:
    prediction_strings = pickle.load(f)
with open(dir_name+'predicted_y', 'rb') as f:
    predicted_y = pickle.load(f)
dict_results_loader = torch.load(dir_name+'/dict_results_loader.pt')

batches = list(range(len(dict_results_loader)))
model.eval()
latents = []
y_p = []
all_reconstructions = []
with torch.no_grad():
    for i, batch in enumerate(batches):
        data = dict_results_loader[str(batch)][0]
        data.to(device)
        dest_is_origin_matrix = dict_results_loader[str(batch)][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_results_loader[str(batch)][2]
        inc_edges_to_atom_matrix.to(device)

        # Perform a single forward pass.
        reconstruction, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
        latents.append(z.cpu().numpy())
        y_p.append(y.cpu().numpy())
        reconstruction_strings = [combine_tokens(tokenids_to_vocab(reconstruction[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(reconstruction))]
        all_reconstructions.extend(reconstruction_strings)

# 4. Compare results with the original output from GA
print(y_p)
print(prediction_strings)
y_p=y_p[0].tolist()
print(y_p)
print(len(y_p))
print(len(prediction_strings))
predicted_y = [tensor.squeeze().tolist() for tensor in predicted_y]
predicted_y = [item for sublist in predicted_y for item in sublist]
i=0
with open(dir_name+'results_optimization.txt', 'w') as f:
    f.write("The final population's best results are the following\n")
    for l1,l2 in zip(all_reconstructions,prediction_strings):
        if l1 == l2:
            f.write("decoded molecule from GA selection is encoded and decoded to the same molecule\n")
            f.write(l1 + "\n")
            f.write("The predicted properties from z(GA) are: " + str(predicted_y[i]) + ". The predicted properties from z(reencoded) are: " + str(y_p[i]) + "\n")
        else:
            f.write("GA molecule: ")
            f.write(l2 + "\n")
            f.write("Encoded and decoded GA molecule: ")
            f.write(l1 + "\n")
            f.write("The predicted properties from z(GA) are: " + str(predicted_y[i]) + ". The predicted properties from z(reencoded) are: " + str(y_p[i]) + "\n")
        i+=1


#Plot the pareto front
import matplotlib.pyplot as plt

F_end = res.F  # Get the objective function values from the optimization result

# Plot EA vs IP
with open(dir_name+'y1_all_'+dataset_type+'.npy', 'rb') as f:
    y1_all = np.load(f)
with open(dir_name+'y2_all_'+dataset_type+'.npy', 'rb') as f:
    y2_all = np.load(f)

with open(dir_name+'yp_all_'+dataset_type+'.npy', 'rb') as f:
    yp_all = np.load(f)
yp1_all = [yp[0] for yp in yp_all]
yp2_all = [yp[1] for yp in yp_all]



# Plot the Pareto front
plt.rcParams.update({'font.size': 16})
y_p_np = np.array(y_p)
plt.figure(figsize=(8, 6))
#plt.scatter(F_end[:, 0], F_end[:, 1], color='blue', marker='o', label='Pareto Front')
plt.scatter(y_p_np[:, 0], y_p_np[:, 1], color='red', marker='D', s=5, label='GA optimized molecules')
plt.scatter(y1_all, y2_all, s=2, alpha=0.5, color='blue', label='training data')
#plt.scatter(yp1_all, yp2_all, s=1, alpha=0.2, color='blue', marker='o', label='training data (augmented)')
plt.xlabel('EA in eV')
plt.ylabel('IP in eV')
plt.title('Train set polymers and novel polymers by GA')
plt.legend()
plt.savefig(dir_name+'paretofront.png',  dpi=300)






""" 

# decode all solutions
prediction_strings= []
predicted_y = []
with torch.no_grad():
    for b in all_solutions:
        bi = torch.from_numpy(b[0]).to(device).to(torch.float32) 
        predictions, _, _, _, y = model.inference(data=bi, device=device, sample=False, log_var=None)
        prediction_strings.append(combine_tokens(tokenids_to_vocab(predictions[0][0].tolist(), vocab), tokenization=tokenization))
        predicted_y.append(y)
        print(prediction_strings[-1], predicted_y[-1])

with open(dir_name+'prediction_strings_all', 'wb') as f:
    pickle.dump(prediction_strings, f)
with open(dir_name+'predicted_y_all', 'wb') as f:
    pickle.dump(predicted_y, f)


# 1. convert the molecular strings to graphs again
# TODO: non valid strings need to be captured somehow
data_list = []
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

# Save the graphs in Dataloader and convert 
results_loader = DataLoader(dataset=data_list, batch_size=64, shuffle=False)
dict_results_loader = MP_Matrix_Creator(results_loader)
torch.save(dict_results_loader, dir_name+'/dict_results_loader_all.pt')

# 2. Encode + predict (forward pass) in eval mode (z=mean) + do inference decoding
#  Run over all epoch

# Start from here if optimization already run before

with open(dir_name+'res_optimization', 'rb') as f:
    res = pickle.load(f)
with open(dir_name+'all_solutions', 'rb') as f:
    best_solutions = pickle.load(f)
with open(dir_name+'prediction_strings_all', 'rb') as f:
    prediction_strings = pickle.load(f)
with open(dir_name+'predicted_y_all', 'rb') as f:
    predicted_y = pickle.load(f)
dict_results_loader = torch.load(dir_name+'/dict_results_loader_all.pt')

batches = list(range(len(dict_results_loader)))
model.eval()
latents = []
y_p = []
all_reconstructions = []
with torch.no_grad():
    for i, batch in enumerate(batches):
        data = dict_results_loader[str(batch)][0]
        data.to(device)
        dest_is_origin_matrix = dict_results_loader[str(batch)][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_results_loader[str(batch)][2]
        inc_edges_to_atom_matrix.to(device)

        # Perform a single forward pass.
        reconstruction, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
        latents.append(z.cpu().numpy())
        y_p.append(y.cpu().numpy())
        reconstruction_strings = [combine_tokens(tokenids_to_vocab(reconstruction[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(reconstruction))]
        all_reconstructions.extend(reconstruction_strings)

# 4. Compare results with the original output from GA
y_p=y_p[0].tolist()
predicted_y = [tensor.squeeze().tolist() for tensor in predicted_y]
i=0
with open(dir_name+'results_optimization_all.txt', 'w') as f:
    f.write("The whole populations are the following\n")
    for l1,l2 in zip(all_reconstructions,prediction_strings):
        if l1 == l2:
            f.write("decoded molecule from GA selection is encoded and decoded to the same molecule\n")
            f.write(l1 + "\n")
            f.write("The predicted properties from z(GA) are: " + str(y_p[i]) + ". The predicted properties from z(reencoded) are: " + str(predicted_y[i]) + "\n")
        else:
            f.write("GA molecule: ")
            f.write(l2 + "\n")
            f.write("Encoded and decoded GA molecule: ")
            f.write(l1 + "\n")
            f.write("The predicted properties from z(GA) are: " + str(y_p[i]) + ". The predicted properties from z(reencoded) are: " + str(predicted_y[i]) + "\n")
        i+=1

 """