import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *
from data_processing.rdkit_poly import *
from data_processing.Smiles_enum_canon import SmilesEnumCanon

# deep learning packages
import torch
import pickle
import argparse


# Necessary lists
all_predictions = []
all_real = []
prediction_validityA = []
prediction_validityB = []
monA_pred = []
monB_pred = []
monA_true = []
monB_true = []
monomer_weights_predicted = []
monomer_weights_real = []
monomer_con_predicted = []
monomer_con_real = []


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

dataset_type = "test"
data_augment = "old" # new or old
dict_test_loader = torch.load(main_dir_path+'/data/dict_test_loader_'+augment+'_'+tokenization+'.pt')


num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features


# Load model
# Create an instance of the G2S model from checkpoint
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
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

    print(f'Run inference')

    #  Run over all epoch
    batches = list(range(len(dict_test_loader)))
    vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    vocab = load_vocab(vocab_file=vocab_file)

### INFERENCE v2: reconstruction is measured by sampling 1000 times and reporting percentage of correct decodings ###
    sm_can = SmilesEnumCanon()
    reconstruction_b = []
    for batch in batches:    
        all_predictions_v2 = []
        all_predictions_mean = []

        with torch.no_grad():
        
            model.eval()
            data = dict_test_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_test_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)
            model.beta =1.0
            real_strings = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization) for sample in range(len(data))]
            predictions, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
            prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
            all_predictions_mean.append(prediction_strings)
            for i in range(100):
                # only for first batch 0
                predictions, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=True, log_var=None)
                prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
                all_predictions_v2.append(prediction_strings)
                
                #reconstructed_SmilesB = list(map(Chem.MolToSmiles, [mon[1] for mon in prediction_validity]))
            # transpose lists
            all_predictions_v2=list(map(list, zip(*all_predictions_v2)))

        # Canonicalize both the prediction and real string and check if they are the same
        percentage_equal_entries=[]    
        for i,reconstructions in enumerate(all_predictions_v2):
            # Count the number of elements equal to real molecule
            all_reconstructions=[s.split('_', 1)[0] for s in reconstructions]
            reconstrucions_can = list(map(sm_can.canonicalize, all_reconstructions))
            ground_truth= real_strings[i].split('_', 1)[0]
            # If there is still the old error in the trained model, correct it 
            substring_to_replace = "4-4:0.375:0.375"
            replacement_substring = "4-4:0.125:0.125"
            # Replace the substring if it's present
            ground_truth = ground_truth.replace(substring_to_replace, replacement_substring)

            ground_truth_can = sm_can.canonicalize(ground_truth)
            count_equal = reconstrucions_can.count(ground_truth_can)
            
            # Calculate the percentage of equal entries
            percentage_equal = (count_equal / len(reconstructions)) * 100
            
            # Append the percentage to the result list
            percentage_equal_entries.append(percentage_equal)

        # For this batch calulate the 
        reconstruction_b.append(sum(percentage_equal_entries)/len(percentage_equal_entries))
        print('Reconstructed batch %d/%d'%(batch, len(batches)))
        # How many batches of the test set should be checked like that
        if batch>7:break


    with open(dir_name+'reconstruction_v2.txt', 'w') as f:
        f.write("Full rec: %.4f %%" % (sum(reconstruction_b)/len(reconstruction_b)))
