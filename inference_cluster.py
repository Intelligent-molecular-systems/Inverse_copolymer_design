from G2S_oldversion import *
# deep learning packages
import torch
from data_utils import *
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
#device = torch.device("cpu")
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

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

dataset_type = "test" 
data_augment = "old" # new or old
dict_test_loader = torch.load('dataloaders_'+data_augment+'augment/dict_test_loader_'+augment+'_'+tokenization+'.pt')

num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features

# Load model
# Create an instance of the G2S_PP model from checkpoint
filepath=os.path.join(os.getcwd(),'Checkpoints_new/Model_onlytorchseed_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/model_best_loss.pt')
if os.path.isfile(filepath):
    if args.ppguided:
        model_type = G2S_PPguided
    else: 
        model_type = G2S_Gab
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model_config = checkpoint["model_config"]
    batch_size = model_config['batch_size']
    hidden_dimension = model_config['hidden_dimension']
    embedding_dimension = model_config['embedding_dim']
    vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')
    if model_config['loss']=="wce":
        vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
        class_weights = token_weights(vocab_file)
        class_weights = torch.FloatTensor(class_weights)
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, loss_weights=class_weights, add_latent=add_latent)
    else: model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, add_latent=add_latent)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Directory to save results
    # Directory to save results
    dir_name= 'Checkpoints_new/Model_onlytorchseed_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f'Run inference')

    #  Run over all epoch
    batches = list(range(len(dict_test_loader)))

    ### INFERENCE ###
    with torch.no_grad():
    # only for first batch
        model.eval()
        for batch in batches:
            data = dict_test_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_test_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)
            model.beta =1.0
            predictions, _, _, z = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
            vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')
            # save predicitons of first validation batch in text file
            prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
            all_predictions.extend(prediction_strings)
            real_strings = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization) for sample in range(len(data))]
            all_real.extend(real_strings)
            #reconstructed_SmilesB = list(map(Chem.MolToSmiles, [mon[1] for mon in prediction_validity]))


            # Evaluation of validation set reconstruction accuracy (inference)
            monomer_smiles_true = [poly_smiles.split("|")[0].split('.') for poly_smiles in real_strings] 
            monomer_smiles_predicted = [poly_smiles.split("|")[0].split('.') for poly_smiles in prediction_strings]
            
            monA_pred.extend([mon[0] for mon in monomer_smiles_predicted])
            try:
                monB_pred.extend([mon[1] for mon in monomer_smiles_predicted])
            except:
                print(monomer_smiles_predicted)
                monB_pred.extend([""]) #In case there is no monomer B predicted
            monA_true.extend([mon[0] for mon in monomer_smiles_true])
            monB_true.extend([mon[1] for mon in monomer_smiles_true])
            

            monomer_weights_predicted.extend([poly_smiles.split("|")[1:-1] for poly_smiles in prediction_strings])
            monomer_weights_real.extend([poly_smiles.split("|")[1:-1] for poly_smiles in real_strings])
            monomer_con_predicted.extend([poly_smiles.split("|")[-1].split("_")[0] for poly_smiles in prediction_strings])
            monomer_con_real.extend([poly_smiles.split("|")[-1].split("_")[0] for poly_smiles in real_strings])

    print(f'Saving inference results')

    with open(dir_name+'all_val_prediction_strings.pkl', 'wb') as f:
        pickle.dump(all_predictions, f)
    with open(dir_name+'all_val_real_strings.pkl', 'wb') as f:
        pickle.dump(all_real, f)
