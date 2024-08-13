from scipy.stats import ortho_group
import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *
from data_processing.Function_Featurization_Own import poly_smiles_to_graph

# deep learning packages
import torch
from torch_geometric.loader import DataLoader
import pickle
import argparse
import random
import numpy as np


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

# Function to generate random orthogonal unit vectors
def random_orthogonal_unit_vectors(mu, logvar, num_vectors, num_std_devs):
    # Compute standard deviation from logvar
    sigma = torch.exp(0.5 * logvar)
    
    # Define the range of the orthogonal vectors
    range_of_vectors = num_std_devs * sigma
    
    # Initialize list to store orthogonal vectors
    vectors = []
    for _ in range(num_vectors):
        # Generate random orthogonal matrix
        orthogonal_matrix = ortho_group.rvs(dim= 2)
        
        # Convert orthogonal_matrix to PyTorch tensor
        orthogonal_matrix = torch.tensor(orthogonal_matrix, dtype=torch.float32, device=device)
        
        # Scale the orthogonal matrix by the range
        scaled_orthogonal_matrix = orthogonal_matrix * range_of_vectors.unsqueeze(0)
        
        # Append the first row of the scaled orthogonal matrix as the vector
        vector = scaled_orthogonal_matrix[0]
        vectors.append(vector)
    
    return torch.stack(vectors, dim=0)

# Function to generate the grid in the latent space
def generate_grid_with_center(z, orthogonal_vectors, grid_size):
    grid = []
    half_size = grid_size // 2
    for i in range(-half_size, half_size + 1):
        for j in range(-half_size, half_size + 1):
            point = z + i * orthogonal_vectors[0] + j * orthogonal_vectors[1]
            grid.append(point)
    return torch.stack(grid)

# lists
all_predictions = []

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


dataset_type = "test"
data_augment = "old" # new or old
dict_test_loader = torch.load(main_dir_path+'/data/dict_test_loader_'+augment+'_'+tokenization+'.pt')


num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features

# Load model
# Create an instance of the G2S model from checkpoint
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
filepath = os.path.join(main_dir_path,'Checkpoints/', model_name,"model_best_loss.pt")
if os.path.isfile(filepath):
    if args.ppguided:
        model_type = G2S_VAE_PPguided
    else: 
        model_type = G2S_VAE_PPguideddisabled
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model_config = checkpoint["model_config"]
    model_config["max_alpha"]=args.max_alpha
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

""" 
    # Generate grid around molecule 
    
    data_list = []
    seed_smiles = "[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2.[*:3]c1cccc2c1sc1c([*:4])cccc12|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5"
    g = poly_smiles_to_graph(seed_smiles, np.nan, np.nan, None)
    target_tokens = tokenize_poly_input_RTlike(poly_input=seed_smiles)
    tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=target_tokens, vocab=vocab)
    g.tgt_token_ids = tgt_token_ids
    g.tgt_token_lens = tgt_lens
    g.to(device)
    data_list.append(g)
    data_loader = DataLoader(dataset=data_list, batch_size=64, shuffle=False)
    dict_data_loader = MP_Matrix_Creator(data_loader)

    all_predictions_seed= []
    with torch.no_grad():
    # only for first batch
        model.eval()

        data = dict_data_loader["0"][0]
        data.to(device)
        dest_is_origin_matrix = dict_data_loader["0"][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_data_loader["0"][2]
        inc_edges_to_atom_matrix.to(device)
        _, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)

        seed_z = z[0]
        print(seed_z)


        # encode a batch to mu and logvar
        h_G_mean, h_G_var = model.Encoder(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        if not model.hidden_dim==model.embedding_dim:
            h_G_mean = model.lincompress(h_G_mean)
            h_G_var = model.lincompress(h_G_var)
        print(h_G_mean)

        # Generate two random orthogonal unit vectors in latent space
        orthogonal_vectors = random_orthogonal_unit_vectors(mu=h_G_mean, logvar=h_G_var, num_vectors=2, num_std_devs=3)

        # Define grid size
        grid_size = 10

        # Generate grid in latent space around seed molecule
        grid = generate_grid_with_center(seed_z, orthogonal_vectors, grid_size)


        # Decode each point in the grid
        decoded_molecules = []
        for point in grid:
            # Add the noise to the original tensor
            decoded_molecules.append(point.cpu().numpy())
            predictions_seed, _, _, z, y = model.inference(data=point, device=device, sample=False, log_var=None)
            prediction_strings = [combine_tokens(tokenids_to_vocab(predictions_seed[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions_seed))]
            decoded_molecules.extend(prediction_strings)
            seed_string = combine_tokens(tokenids_to_vocab(data.tgt_token_ids[0], vocab),tokenization=tokenization)
            print(prediction_strings)


    print(f'Saving generated strings')

    #with open(dir_name+'generated_polymers.pkl', 'wb') as f:
    #    pickle.dump(all_predictions, f)

    #with open(dir_name+'generated_polymers.pkl', 'wb') as f:
    #    pickle.dump(all_predictions, f)
    with open(dir_name+'seed_literature_grid.txt', 'w') as f:
        f.write("Seed molecule: %s " %seed_string)
        f.write("The following are the generations from seed (mean) with noise\n")
        for s in decoded_molecules:
            f.write(f"{s}\n")
    with open(dir_name+'seed_literature_grid_latents.npy', 'wb') as f:
        sampled_z = np.stack(grid)
        #print(sampled_z)
        np.save(f, sampled_z)
    with open(dir_name+'seed_literature_latent.npy', 'wb') as f:
        seed_z = seed_z.cpu().numpy()
        np.save(f, seed_z)
 """
    
## Interpolation between two best molecules of Bai et. al
with torch.no_grad():
    # only for first batch
    all_predictions_interp = []
    data_list = []
    model.eval()    
    start_mol = "[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2.[*:3]c1cccc2c1sc1c([*:4])cccc12|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5"
    end_mol = "[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2.[*:3]c1ccc2c3ccc([*:4])cc3c3ccccc3c2c1|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5"

    for mol in [start_mol, end_mol]:
        g = poly_smiles_to_graph(mol, np.nan, np.nan, None)
        target_tokens = tokenize_poly_input_RTlike(poly_input=mol)
        tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=target_tokens, vocab=vocab)
        g.tgt_token_ids = tgt_token_ids
        g.tgt_token_lens = tgt_lens
        g.to(device)
        data_list.append(g)
    data_loader = DataLoader(dataset=data_list, batch_size=64, shuffle=False)
    dict_data_loader = MP_Matrix_Creator(data_loader)

    all_predictions_seed= []
    with torch.no_grad():
    # only for first batch
        model.eval()

        data = dict_data_loader["0"][0]
        data.to(device)
        dest_is_origin_matrix = dict_data_loader["0"][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_data_loader["0"][2]
        inc_edges_to_atom_matrix.to(device)
        _, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)

        seed_z1 = z[0].to(device)
        seed_z2 = z[1].to(device)
        print(seed_z1, seed_z2)

        print(start_mol, end_mol)

        # Number of steps for interpolation
        num_steps = 10

        # Calculate the step size for each dimension
        step_sizes = (seed_z2 - seed_z1) / (num_steps + 1)  # Adding 1 to include the endpoints

        # Generate interpolated vectors
        interpolated_vectors = [seed_z1 + i * step_sizes for i in range(1, num_steps + 1)]

        # Include the endpoints
        interpolated_vectors = torch.stack([seed_z1]+ interpolated_vectors+ [seed_z2])

        # Display the interpolated vectors
        for s in range(interpolated_vectors.shape[0]):
            prediction_interp, _, _, _, y = model.inference(data=interpolated_vectors[s], device=device, sample=False, log_var=None)
            prediction_string = combine_tokens(tokenids_to_vocab(prediction_interp[0][0].tolist(), vocab), tokenization=tokenization)
            all_predictions_interp.append(prediction_string)

        cwd = os.getcwd()
        with open(cwd+'interpolated_polymers_between_best.txt', 'w') as f:
            f.write("Molecule1: %s \n" %start_mol)
            f.write("Molecule2: %s \n" %end_mol)
            f.write("The following are the stepwise interpolated molecules\n")
            for s in all_predictions_interp:
                f.write(f"{s}\n")