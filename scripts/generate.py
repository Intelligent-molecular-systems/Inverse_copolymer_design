# %% Packages
import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *

# deep learning packages
import torch
import pickle
import argparse
import random
import numpy as np

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

    # Directory to save results
    dir_name=  os.path.join(main_dir_path,'Checkpoints/', model_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


    print(f'Generate random samples')

    ### INFERENCE ###
    torch.manual_seed(args.seed)
    with torch.no_grad():
    ## generate samples from random noise
    # only for first batch
        model.eval()
        for i in range(250):
            z_rand = torch.randn((64,embedding_dimension), device=device)*args.epsilon
            print(z_rand.shape)
            predictions_rand, _, _, z, y = model.inference(data=z_rand, device=device, sample=False, log_var=None)
            print('Generated batch %d/250'%i)
            # save predicitons of first validation batch in text file
            prediction_strings = [combine_tokens(tokenids_to_vocab(predictions_rand[sample][0].tolist(), vocab), tokenization=tokenization).split('_')[0] for sample in range(len(predictions_rand))]
            all_predictions.extend(prediction_strings)
   
    with open(dir_name+'generated_polymers.pkl', 'wb') as f:
        pickle.dump(all_predictions, f)

    ## generate samples around seed molecule

    all_predictions_seed = []
    batches = list(range(len(dict_test_loader)))
    #randomly select batch
    random.seed(args.seed)
    batch = random.choice(batches)
    with torch.no_grad():
    # only for first batch
        model.eval()

        data = dict_test_loader[str(batch)][0]
        data.to(device)
        dest_is_origin_matrix = dict_test_loader[str(batch)][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
        inc_edges_to_atom_matrix.to(device)
        _, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
        #seed_strings = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[ind], vocab),tokenization=tokenization) for ind in range(64)]
        #print(seed_strings)
        # randomly select a seed molecule
        ind = random.choice(list(range(64)))
        seed_z = z[ind]
        print(seed_z)
        seed_z = seed_z.unsqueeze(0).repeat(64,1)
        print(seed_z[0])
        sampled_z = []
        for r in range(8):
            # Define the mean and standard deviation of the Gaussian noise
            mean = 0
            std = args.epsilon/2 #half of epsilon
            # Create a tensor of the same size as the original tensor with random noise
            print(seed_z)
            print(seed_z.size())
            noise = torch.tensor(np.random.normal(mean, std, size=seed_z.size()), dtype=torch.float, device=device)

            # Add the noise to the original tensor
            seed_z_noise = seed_z + noise
            sampled_z.append(seed_z_noise.cpu().numpy())
            predictions_seed, _, _, z, y = model.inference(data=seed_z_noise, device=device, sample=False, log_var=None)
            prediction_strings = [combine_tokens(tokenids_to_vocab(predictions_seed[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions_seed))]
            all_predictions_seed.extend(prediction_strings)
            seed_string = combine_tokens(tokenids_to_vocab(data.tgt_token_ids[ind], vocab),tokenization=tokenization)


    print(f'Saving generated strings')

    #with open(dir_name+'generated_polymers.pkl', 'wb') as f:
    #    pickle.dump(all_predictions, f)
    with open(dir_name+'seed_polymer.txt', 'w') as f:
        f.write('%s'%seed_string)

    #with open(dir_name+'generated_polymers.pkl', 'wb') as f:
    #    pickle.dump(all_predictions, f)
    with open(dir_name+'seed_polymers_noise'+str(std)+'.txt', 'w') as f:
        f.write("Seed molecule: %s " %seed_string)
        f.write("The following are the generations from seed (mean) with noise\n")
        for s in all_predictions_seed:
            f.write(f"{s}\n")
    with open(dir_name+'seed_polymers_latents_noise'+str(std)+'.npy', 'wb') as f:
        print(sampled_z)
        sampled_z = np.stack(sampled_z)
        #print(sampled_z)
        np.save(f, sampled_z)
    with open(dir_name+'seed_polymer_z.npy', 'wb') as f:
        seed_z = seed_z.cpu().numpy()
        np.save(f, seed_z)
    with open(dir_name+'generated_polymers_from_seed_noise'+str(std)+'.pkl', 'wb') as f:
        pickle.dump(all_predictions_seed, f)


    #Interpolation
    ## generate samples interpolated between two seed molecules
    dataset_type = "train"
    data_augment = "old" # new or old
    dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')


    all_predictions_interp = []
    batches = list(range(len(dict_test_loader)))
    #randomly select batch
    random.seed(args.seed)
    batch = random.choice(batches)
    with torch.no_grad():
    # only for first batch
        model.eval()

        data = dict_test_loader[str(batch)][0]
        data.to(device)
        dest_is_origin_matrix = dict_test_loader[str(batch)][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
        inc_edges_to_atom_matrix.to(device)
        _, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
        #seed_strings = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[ind], vocab),tokenization=tokenization) for ind in range(64)]
        #print(seed_strings)
        # randomly select a seed molecule
        
        examples = 10
        for e in range(examples):
            all_predictions_interp = []
            ind1 = random.choice(list(range(64)))
            ind2 = random.choice(list(range(64)))
            assert not (ind1==ind2)
            start_mol = combine_tokens(tokenids_to_vocab(data.tgt_token_ids[ind1], vocab),tokenization=tokenization)
            end_mol = combine_tokens(tokenids_to_vocab(data.tgt_token_ids[ind2], vocab),tokenization=tokenization)
            print(z.shape)
            seed_z1 = z[ind1]
            seed_z2 = z[ind2]
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

            with open(dir_name+'interpolated_polymers_example'+str(e)+'.txt', 'w') as f:
                f.write("Molecule1: %s \n" %start_mol)
                f.write("Molecule2: %s \n" %end_mol)
                f.write("The following are the stepwise interpolated molecules\n")
                for s in all_predictions_interp:
                    f.write(f"{s}\n")

else: print("The model training diverged and there are is no trained model file!")