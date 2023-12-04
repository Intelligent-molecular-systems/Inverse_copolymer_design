from G2S import *
# deep learning packages
import torch
from data_utils import *
import pickle
import argparse
import re

# TODO: make this more elegant
# for now we convert the strings back to the old string so we can use the old scripts for metrics etc. 
# Future: make it work for all numbers
def convert_to_oldstring(s):
    print(s)
    pattern = r'\[\*\|(.*?)\|\]'
    substrings = re.findall(pattern, s)
    weights= {i+1:w for i,w in enumerate(substrings)}

    counter_conversion = 1 # set global counter to 0
    # Define the regex pattern to match substrings like [*|0.5|0.5|0.0|0.0|]
    pattern = r'\[\*\|(.*?)\|\]'
    # Replace matches using the `replacement` function
    newstring = ''
    start = 0
    for sub in re.finditer(pattern, s):
        end, newstart = sub.span()
        newstring += s[start:end]
        rep = "[*:{}]".format(counter_conversion)
        newstring += rep
        start = newstart
        counter_conversion+=1
    newstring += s[start:]
    print(weights)

    # the three connectivity cases
    if " ".join(list(weights.values())) == "0.25|0.25|0.25|0.25 0.25|0.25|0.25|0.25 0.25|0.25|0.25|0.25 0.25|0.25|0.25|0.25":
        newstring+="|<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25"
    elif " ".join(list(weights.values())) == '0.5|0.5|0.0|0.0 0.5|0.5|0.0|0.0 0.0|0.0|0.5|0.5 0.0|0.0|0.5|0.5':
        newstring+="|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5"
    elif " ".join(list(weights.values())) == '0.375|0.375|0.125|0.125 0.375|0.375|0.125|0.125 0.125|0.125|0.375|0.375 0.125|0.125|0.375|0.375':
        newstring+="|<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125"
    pattern = r'\|\|'
    # Replace matches using the `replacement` function
    return re.sub(pattern, "|", newstring)



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


args = parser.parse_args()

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "test"
dict_test_loader = torch.load('dict_'+dataset_type+'_loader_'+augment+'_'+tokenization+'_gbigsmileslike.pt')

num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features

# Load model
# Create an instance of the G2S_PP model from checkpoint
filepath=os.path.join(os.getcwd(),'Checkpoints_new/Model_beta='+str(args.beta)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_gbigsmileslike/model_best_loss.pt')
if os.path.isfile(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model_config = checkpoint["model_config"]
    batch_size = model_config['batch_size']
    hidden_dimension = model_config['hidden_dimension']
    embedding_dimension = model_config['embedding_dim']
    vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'_gbigsmileslike.txt')
    if model_config['loss']=="wce":
        vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'_gbigsmileslike.txt'
        class_weights = token_weights(vocab_file)
        class_weights = torch.FloatTensor(class_weights)
        model = G2S_Gab(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, loss_weights=class_weights, add_latent=add_latent)
    else: model = G2S_Gab(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, add_latent=add_latent)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Directory to save results
    # Directory to save results
    dir_name= 'Checkpoints_new/Model_beta='+str(args.beta)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_gbigsmileslike/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print(f'Run inference')

    #  Run over all epoch
    batches = list(range(len(dict_test_loader)))
    print(len(dict_test_loader))

    ### INFERENCE ###
    with torch.no_grad():
    # only for first batch
        model.eval()
        for batch in batches:
            batch =67
            data = dict_test_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_test_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_test_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)
            model.beta =1.0
            predictions, _, _, z = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
            vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'_gbigsmileslike.txt')
            # save predicitons of first validation batch in text file
            prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
            prediction_strings = [convert_to_oldstring(s) for s in prediction_strings]
            print("real")
            all_predictions.extend(prediction_strings)
            real_strings = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization) for sample in range(len(data))]
            real_strings = [convert_to_oldstring(s) for s in real_strings]
            all_real.extend(real_strings)
            if batch>0: break
            

    print(f'Saving inference results')

    with open(dir_name+'all_val_prediction_strings.pkl', 'wb') as f:
        pickle.dump(all_predictions, f)
    with open(dir_name+'all_val_real_strings.pkl', 'wb') as f:
        pickle.dump(all_real, f)
