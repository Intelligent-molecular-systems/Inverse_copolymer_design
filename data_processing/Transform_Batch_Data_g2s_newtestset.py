# %% Packages
import os, sys
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

import numpy as np
import torch
from torch.utils.data import Dataset
from data_processing.Function_Featurization_Own import poly_smiles_to_graph
from data_processing.data_utils import *
import pandas as pd
import networkx as nx
from torch_geometric.utils import to_networkx
import random
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import argparse


# %% Hyperparameters
device = 'cpu'
# %% Call data
parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original, augmented_canonical", default="augmented", choices=["augmented", "original", "augmented_canonical", "augmented_enum"])
parser.add_argument("--batch_size", type=int, default=64)


args = parser.parse_args()

augment = args.augment
batch_size = args.batch_size

tokenization = "RT_tokenized"
string_format = "poly_chemprop" # "poly_chemprop" or "gbigsmileslike"
smiles_enumeration = True
if augment == "augmented_enum":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-enumerated2-poly_chemprop.csv')
elif augment == "original":
    df = pd.read_csv(main_dir_path+'/data/dataset-poly_chemprop.csv')
elif augment == "original_canonical":
    df = pd.read_csv(main_dir_path+'/data/dataset-poly_chemprop_canonical.csv')
elif augment == "augmented_canonical":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-canonical-poly_chemprop.csv')
elif augment == "augmented":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-poly_chemprop_v2.csv')
# %% Lets create PyG data objects

# uncomment if graphs_list.pt does not exist
# Here we turn all smiles tring and featurize them into graphs and put them in a list: graphs_list
# additionally we add the target token ids of the target string as graph attributes 

Graphs_list = []
target_tokens_list = []
target_tokens_ids_list = []
target_tokens_lens_list = []
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    try: poly_input_nocan = df.loc[i, 'poly_chemprop_input_nocan']
    except: poly_input_nocan=None
    poly_label1 = df.loc[i, 'EA vs SHE (eV)']
    poly_label2 = df.loc[i, 'IP vs SHE (eV)']
    graphs = poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan)
    #if string_format == "gbigsmileslike":
    #    poly_input_gbigsmileslike = df.loc[i, 'poly_chemprop_input_GbigSMILESlike']
    #    target_tokens = tokenize_poly_input_new(poly_input=poly_input_gbigsmileslike, tokenization=tokenization)
    #elif string_format=="poly_chemprop":
    if tokenization=="oldtok":
        target_tokens = tokenize_poly_input(poly_input=poly_input)
    elif tokenization=="RT_tokenized":
        target_tokens = tokenize_poly_input_RTlike(poly_input=poly_input)
    Graphs_list.append(graphs)
    target_tokens_list.append(target_tokens)
    if i % 100 == 0:
        print(f"[{i} / {len(df.loc[:, 'poly_chemprop_input'])}]")

# Create vocab file 
make_vocab(target_tokens_list=target_tokens_list, vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')


# convert the target_tokens_list to target_token_ids_list using the vocab file
# load vocab dict (token:id)
vocab = load_vocab(vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')
max_tgt_token_length = len(max(target_tokens_list, key=len))
for tgt_tokens in target_tokens_list:
    tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=tgt_tokens, vocab=vocab, max_tgt_len=max_tgt_token_length)
    target_tokens_ids_list.append(tgt_token_ids)
    target_tokens_lens_list.append(tgt_lens)

# Add the tgt_token_ids as additional attributes in graph
for sample_idx, g in enumerate(Graphs_list): 
    g.tgt_token_ids = target_tokens_ids_list[sample_idx]
    g.tgt_token_lens = target_tokens_lens_list[sample_idx]

# Save graphs (and tgt token) data
torch.save(Graphs_list, main_dir_path+'/data/Graphs_list_'+augment+'_'+tokenization+'.pt')

# Create training, self supervised and test sets

# shuffle graphs
# we first take out the validation and test set from 
random.seed(12345)
if augment == "original" or augment=="original_canonical":
    # new improved test set: exclude monomer combinations completely
    mon_combs = []
    monB_list = []
    stoichiometry_connectivity_combs = []
    for i in range(len(df.loc[:, 'poly_chemprop_input'])):
        poly_input = df.loc[i, 'poly_chemprop_input']
        poly_label1 = df.loc[i, 'EA vs SHE (eV)']
        poly_label2 = df.loc[i, 'IP vs SHE (eV)']
        mon_combs.append(".".join(poly_input.split("|")[0].split('.')))

    mon_combs= list(set(mon_combs))
    mon_combs_shuffle = random.sample(mon_combs, len(mon_combs))
    # take 80-20 split for trainig -  test data
    train_mon_combs = mon_combs_shuffle[:int(0.8*len(mon_combs_shuffle))]
    val_mon_combs = mon_combs_shuffle[int(0.8*len(mon_combs_shuffle)):int(0.9*len(mon_combs_shuffle))]
    test_mon_combs = mon_combs_shuffle[int(0.9*len(mon_combs_shuffle)):]

    
    #Go through graphs list and assign 
    Graphs_list = torch.load(main_dir_path+'/data/Graphs_list_original'+'_'+tokenization+'.pt')
    data_list_shuffle = random.sample(Graphs_list, len(Graphs_list))
    train_datalist=[]
    val_datalist=[]
    test_datalist=[]
    for graph in Graphs_list: 
        if ".".join(graph.monomer_smiles) in train_mon_combs:
            train_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in val_mon_combs:
            val_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in test_mon_combs:
            test_datalist.append(graph)

    # print some statistics
    print(f'Number of training graphs: {len(train_datalist)}')
    print(f'Number of test graphs: {len(test_datalist)}')

elif augment == "augmented_enum" or augment == "augmented_canonical":
    Graphs_list = torch.load(main_dir_path+'/data/Graphs_list_original'+'_'+tokenization+'.pt')
    Graphs_list_combined = torch.load(main_dir_path+'/data/Graphs_list_'+augment+'_'+tokenization+'.pt')
    org_polymers = Graphs_list_combined[:len(Graphs_list)]
    augm_polymers = Graphs_list_combined[len(Graphs_list):] 
    mon_combs=[]
    # go through original data
    for graph in org_polymers:
        if augment=="augmented_canonical" or augment=="augmented_enum":
            mon_combs.append(".".join(graph.monomer_smiles_nocan))
        else: 
            mon_combs.append(".".join(graph.monomer_smiles))

    mon_combs= list(set(mon_combs))
    mon_combs_shuffle = random.sample(mon_combs, len(mon_combs))
    # take 80-20 split for trainig -  test data
    train_mon_combs = mon_combs_shuffle[:int(0.8*len(mon_combs_shuffle))]
    val_mon_combs = mon_combs_shuffle[int(0.8*len(mon_combs_shuffle)):int(0.9*len(mon_combs_shuffle))]
    test_mon_combs = mon_combs_shuffle[int(0.9*len(mon_combs_shuffle)):]

    train_datalist=[]
    val_datalist=[]
    test_datalist=[]
    for graph in org_polymers: 
        if ".".join(graph.monomer_smiles_nocan) in train_mon_combs:
            train_datalist.append(graph)
        elif ".".join(graph.monomer_smiles_nocan) in val_mon_combs:
            val_datalist.append(graph)
        elif ".".join(graph.monomer_smiles_nocan) in test_mon_combs:
            test_datalist.append(graph)

    # go through the augmented data 
    mon_combs_new=[]
    for graph in augm_polymers:
        if not ".".join(graph.monomer_smiles_nocan) in mon_combs:
            # only monomer combinations that have not been seen in the original dataset
            mon_combs_new.append(".".join(graph.monomer_smiles_nocan))
        # Check that the same monomer combinations are not in train and testset



    mon_combs_augm= list(set(mon_combs_new))
    mon_combs_augm_shuffle = random.sample(mon_combs_augm, len(mon_combs_augm))
    train_mon_combs_augm = mon_combs_augm_shuffle[:int(0.95*len(mon_combs_augm_shuffle))]
    test_mon_combs_augm = mon_combs_augm_shuffle[int(0.95*len(mon_combs_augm_shuffle)):]

    for graph in augm_polymers: 
        if ".".join(graph.monomer_smiles_nocan) in train_mon_combs_augm:
            train_datalist.append(graph)
        elif ".".join(graph.monomer_smiles_nocan) in test_mon_combs_augm:
            test_datalist.append(graph)

elif augment == "augmented":
    Graphs_list = torch.load(main_dir_path+'/data/Graphs_list_original'+'_'+tokenization+'.pt')
    Graphs_list_combined = torch.load(main_dir_path+'/data/Graphs_list_'+augment+'_'+tokenization+'.pt')
    org_polymers = Graphs_list_combined[:len(Graphs_list)]
    augm_polymers = Graphs_list_combined[len(Graphs_list):] 
    mon_combs=[]
    # go through original data
    for graph in org_polymers:
        mon_combs.append(".".join(graph.monomer_smiles))

    mon_combs= list(set(mon_combs))
    mon_combs_shuffle = random.sample(mon_combs, len(mon_combs))
    # take 80-20 split for trainig -  test data
    train_mon_combs = mon_combs_shuffle[:int(0.8*len(mon_combs_shuffle))]
    val_mon_combs = mon_combs_shuffle[int(0.8*len(mon_combs_shuffle)):int(0.9*len(mon_combs_shuffle))]
    test_mon_combs = mon_combs_shuffle[int(0.9*len(mon_combs_shuffle)):]

    train_datalist=[]
    val_datalist=[]
    test_datalist=[]
    for graph in org_polymers: 
        if ".".join(graph.monomer_smiles) in train_mon_combs:
            train_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in val_mon_combs:
            val_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in test_mon_combs:
            test_datalist.append(graph)

    # go through the augmented data 
    mon_combs_new=[]
    for graph in augm_polymers:
        if not ".".join(graph.monomer_smiles) in mon_combs:
            # only monomer combinations that have not been seen in the original dataset
            mon_combs_new.append(".".join(graph.monomer_smiles))
        # Check that the same monomer combinations are not in train and testset



    mon_combs_augm= list(set(mon_combs_new))
    mon_combs_augm_shuffle = random.sample(mon_combs_augm, len(mon_combs_augm))
    train_mon_combs_augm = mon_combs_augm_shuffle[:int(0.95*len(mon_combs_augm_shuffle))]
    test_mon_combs_augm = mon_combs_augm_shuffle[int(0.95*len(mon_combs_augm_shuffle)):]

    for graph in augm_polymers: 
        if ".".join(graph.monomer_smiles) in train_mon_combs_augm:
            train_datalist.append(graph)
        elif ".".join(graph.monomer_smiles) in test_mon_combs_augm:
            test_datalist.append(graph)

    # print some statistics
    print(f'Number of training graphs: {len(train_datalist)}')
    print(f'Number of test graphs: {len(test_datalist)}')

num_node_features = train_datalist[0].num_node_features
num_edge_features = train_datalist[0].num_edge_features
print(f'Number of node feautres: {num_node_features}')
print(f'Numer of edge feautres:{num_edge_features} ')


# %%batch them
train_loader = DataLoader(dataset=train_datalist,
                          batch_size=batch_size, shuffle=True) 
val_loader = DataLoader(dataset=val_datalist,
                         batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_datalist,
                         batch_size=batch_size, shuffle=False)

# check that it works, each batch has one big graph
for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}', '\n')
    print(data)
    print()
    if step == 1:
        break

# %% Create Matrices needed for Message Passing


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
        I = torch.empty(2, 0, dtype=torch.long)
        V = torch.empty(0)

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
            i1 = edge.repeat_interleave(len(incoming_edges_idx))
            i2 = incoming_edges_idx.clone()
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

        I = torch.empty(2, 0, dtype=torch.long)
        V = torch.empty(0)
        for i in range(nodes.shape[0]):
            # find index of edges that are incoming to specific atom
            inc_edges_idx = (edge_index[1] == i).nonzero().flatten()
            weights_inc_edges = bond_weights[inc_edges_idx]
            inc_edges_to_atom_matrix[i, inc_edges_idx] = weights_inc_edges

            # for sparse version
            node = torch.tensor([i])
            i1 = node.repeat_interleave(len(inc_edges_idx))
            i2 = inc_edges_idx.clone()
            i = torch.stack((i1, i2), dim=0)
            v = weights_inc_edges

            I = torch.cat((I, i), dim=1)
            V = torch.cat((V, v))

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


# %% Create dictionary with bathed graphs and message passing matrices for supervised train set

dict_train_loader = MP_Matrix_Creator(train_loader)
torch.save(dict_train_loader, main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')
dict_val_loader = MP_Matrix_Creator(val_loader)
torch.save(dict_val_loader, main_dir_path+'/data/dict_val_loader_'+augment+'_'+tokenization+'.pt')
# %% Create dictionary with bathed graphs and message passing matrices for test set
dict_test_loader = MP_Matrix_Creator(test_loader)
torch.save(dict_test_loader, main_dir_path+'/data/dict_test_loader_'+augment+'_'+tokenization+'.pt')


print('Done')


