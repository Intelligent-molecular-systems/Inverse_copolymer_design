from G2S import *
# deep learning packages
import torch
from data_utils import *
import pickle


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
# Call data
dataset_type = "val"
augment = "augmented"
tokenization = "oldtok"

dict_test_loader = torch.load('dict_'+dataset_type+'_loader_'+augment+'_'+tokenization+'.pt')

num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features

# Load model
# Create an instance of the G2S_PP model from checkpoint
checkpoint = torch.load(os.path.join(os.getcwd(),'Checkpoints/Model_h=300_decL=4_nheads=4_beta=schedule_loss=wce_augmented_oldtok_randomseed_newVAE.pt'), map_location=torch.device('cpu'))
model_config = checkpoint["model_config"]
batch_size = model_config['batch_size']
hidden_dimension = model_config['hidden_dimension']
vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')
if model_config['loss']=="wce":
    vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    class_weights = token_weights(vocab_file)
    class_weights = torch.FloatTensor(class_weights)
else: class_weights=None
model = G2S_Gab(num_node_features,num_edge_features,hidden_dimension,hidden_dimension,device,model_config,vocab, loss_weights=class_weights)
model.load_state_dict(checkpoint['model_state_dict'])
#model.load_state_dict(torch.load('Models_Weights/weights_training='+str(1)+'.pt', map_location=torch.device('cpu')))
model.to(device)

# Directory to save results
dir_name= 'Model_h='+str(model_config['hidden_dimension'])+'_decL='+str(model_config['decoder_num_layers'])+'_nheads='+str(model_config['num_attention_heads'])+'_beta='+str(model_config['beta'])+'_loss='+str(model_config['loss'])+'_augmented_oldtok_randomseed_newVAE'+'/'
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
