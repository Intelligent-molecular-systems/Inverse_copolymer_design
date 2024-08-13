import numpy as np
import torch.optim as optim
import argparse
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
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

class PropertyPrediction():
    def __init__(self, model, nr_vars):
        self.model_predictor = model
        self.weight_electron_affinity = 1  # Adjust the weight for electron affinity
        self.weight_ionization_potential = 1  # Adjust the weight for ionization potential
        self.weight_z_distances = 5  # Adjust the weight for distance between GA chosen z and reencoded z
        self.penalty_value = -5  # Adjust the weight for penalty of validity
        self.results_custom = {}
        self.nr_vars = nr_vars
        self.eval_calls = 0

    def evaluate(self, x):
        #  x is a torch tensor with 32 numerical parameters

        # Inference: forward pass NN prediciton of properties and beam search decoding from latent
        #x = torch.from_numpy(np.array(list(params.values()))).to(device).to(torch.float32)
        self.eval_calls += 1
        x.to(device).to(torch.float32)
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
        #dst = np.array([np.linalg.norm(a - b) for a,b in zip(x.cpu(), expanded_z_p)])
        #print(dst)
        # Use the encoded and predicted properties as evaluation for GA (more realistic property prediction)
        #out["F"] = np.zeros((x.shape[0], 3)) # for three objectives (additionally minimze the distance of z vectors)
        obj1 = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]
        obj2 = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1)
        if validity[0]:
            obj3=0
            aggr_obj = -(obj1[0]+obj2[0]+obj3)
        else:
            obj3 = self.penalty_value
            aggr_obj = obj3
        
        # results
        results_dict = {
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
        all_reconstructions = []
        #with torch.no_grad():
        for i, batch in enumerate(batches):
            data = dict_data_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_data_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_data_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)

            # Perform a single forward pass.
            reconstruction, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
            y_p.append(y.cpu().detach().numpy())
            z_p.append(z.cpu().detach().numpy())
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

bounds = [(j, k) for j,k in zip(min_values,max_values)]


nr_vars = 32
prop_predictor = PropertyPrediction(model, nr_vars)

# Initialize the variables randomly within the bounds
initial_solution = torch.tensor(np.random.uniform([bound[0] for bound in bounds], [bound[1] for bound in bounds], size=(32,)), requires_grad=True, dtype=torch.float32)

# Define the Adam optimizer
optimizer = optim.Adam([initial_solution], lr=0.01)

# Optimization loop
# TODO: Does not work as expected: Gradient computation fails in the topk decoding of inference in transformer decoder
# Hence gradients are none
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    objective_value = torch.from_numpy(np.asarray(prop_predictor.evaluate(initial_solution))).to(device).to(torch.float32)
    objective_value.requires_grad=True
    objective_value.backward()

    # Check if gradient is not None before performing optimization step
    if initial_solution.grad is not None:
        optimizer.step()
        # Manually update the values of initial_solution with the optimized values
        with torch.no_grad():
            initial_solution -= 0.01 * initial_solution.grad

            # Manually zero the gradients after updating the parameters
            initial_solution.grad.zero_()
    else:
        print("Gradient is None, skipping optimization step.")
    print(objective_value)
    print(initial_solution)

# Get the optimized solution
optimized_solution = initial_solution.detach().numpy()
optimized_objective_value = prop_predictor.evaluate(initial_solution).detach().numpy()
results_custom = prop_predictor.results_custom


with open('optimization_results_GB_custom.pkl', 'wb') as f:
    pickle.dump(results_custom, f)