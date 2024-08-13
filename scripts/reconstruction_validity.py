import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from data_processing.data_utils import *
from data_processing.rdkit_poly import *
from data_processing.Smiles_enum_canon import SmilesEnumCanon

import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import pickle
from statistics import mean
import argparse
from functools import partial


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
data_augment ="old"
vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
vocab = load_vocab(vocab_file=vocab_file)

# Directory to save results
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
dir_name = os.path.join(main_dir_path,'Checkpoints/', model_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

print(f'Validity check of validation set using inference decoding')

with open(dir_name+'all_val_prediction_strings.pkl', 'rb') as f:
    all_predictions=pickle.load(f)
with open(dir_name+'all_val_real_strings.pkl', 'rb') as f:
    all_real=pickle.load(f)
# Remove all '_' from the strings (EOS token)
all_predictions=[s.split('_', 1)[0] for s in all_predictions]
all_real=[s.split('_', 1)[0] for s in all_real]


with open(dir_name+'all_val_prediction_strings.txt', 'w') as f:
    for s in all_predictions:
        f.write(s+'\n')
with open(dir_name+'all_val_real_strings.txt', 'w') as f:
    for s in all_real:
        f.write(s+'\n')

# Canonicalize both the prediction and real string and check if they are the same
sm_can = SmilesEnumCanon()
all_predictions_can = list(map(sm_can.canonicalize, all_predictions))
all_real_can = list(map(sm_can.canonicalize, all_real))

prediction_validityA = []
prediction_validityB = []
rec_A = []
rec_B = []
rec = []
rec_stoich = []
rec_con = []

for s_r, s_p in zip(all_real, all_predictions):
    #Both canonicalized strings are the same
    if sm_can.canonicalize(s_r) == sm_can.canonicalize(s_p):
        rec.append(True)
        prediction_validityA.append(True)
        prediction_validityB.append(True)
        rec_A.append(True)
        rec_B.append(True)
        rec_stoich.append(True)
        rec_con.append(True)
    # check all the single elements
    else:
        rec.append(False)
        if len(s_p.split("|")[0].split('.'))>1:
            # only 2 monomers is considered valid 
            monA_r_can=sm_can.canonicalize(s_r.split("|")[0].split('.')[0],monomer_only=True)
            monB_r_can=sm_can.canonicalize(s_r.split("|")[0].split('.')[1],monomer_only=True)
            monA_p_can=sm_can.canonicalize(s_p.split("|")[0].split('.')[0],monomer_only=True)
            monB_p_can=sm_can.canonicalize(s_p.split("|")[0].split('.')[1],monomer_only=True)
            # Monomer A
            if not monA_p_can == 'invalid_monomer_string':
                prediction_validityA.append(True)
                if monA_p_can==monA_r_can:
                    rec_A.append(True)
                else: rec_A.append(False)
            else:
                prediction_validityA.append(False)
                rec_A.append(False)
            # Monomer B
            if not monB_p_can == 'invalid_monomer_string':
                prediction_validityB.append(True)
                if monB_p_can==monB_r_can:
                    rec_B.append(True)
                else: rec_B.append(False)
            else:
                prediction_validityB.append(False)
                rec_B.append(False)
            # Stoichiometry
            if s_p.split("|")[1:-1]==s_r.split("|")[1:-1]:
                rec_stoich.append(True)
            else: rec_stoich.append(False)
            if s_p.split("<")[1:]==s_r.split("<")[1:]:
                rec_con.append(True)
            else: rec_con.append(False)
        else: 
            prediction_validityA.append(False)
            prediction_validityB.append(False)
            rec_A.append(False)
            rec_B.append(False)
            rec_stoich.append(False)
            rec_con.append(False)

# the following for loop only for datasets where monA and monB are not in the ordered position (canonicalized whole polymer strings)
""" for s_r, s_p in zip(all_real_can, all_predictions_can):
    if not s_p == "invalid_polymer_string":
        #if the whole canonical string is the same then everything is correctly reconstructed
        if s_r == s_p:
            rec.append(True)
            prediction_validityA.append(True)
            prediction_validityB.append(True)
            rec_A.append(True)
            rec_B.append(True)
            rec_stoich.append(True)
            rec_con.append(True)

        # check all the single elements
        else:
            prediction_validityA.append(True)
            prediction_validityB.append(True)
            if s_r.split("|")[0]==s_p.split("|")[0]:
                rec_A.append(True)
                rec_B.append(True)
            else: 
                monomerA_B_gen_p = [re.sub(r'\*\:\d+', '*:', s_p.split("|")[0].split('.')[0]), re.sub(r'\*\:\d+', '*:', s_p.split("|")[0].split('.')[1])]
                monomerA_B_gen_r = [re.sub(r'\*\:\d+', '*:', s_r.split("|")[0].split('.')[0]), re.sub(r'\*\:\d+', '*:', s_r.split("|")[0].split('.')[1])]
                matching_dict = {string: 1 if string in monomerA_B_gen_p else 0 for string in monomerA_B_gen_r}
                matching = [matching_dict[key] for key in monomerA_B_gen_r]
                if matching[0]:
                    rec_A.append(True)
                    rec_B.append(False)
                elif matching[1]:
                    rec_A.append(False)
                    rec_B.append(True)
                else: 
                    rec_A.append(False)
                    rec_B.append(False)
            if s_p.split("|")[1:-1]==s_r.split("|")[1:-1]:
                rec_stoich.append(True)
            else: rec_stoich.append(False)
            if s_p.split("<")[1:]==s_r.split("<")[1:]:
                rec_con.append(True)
            else: rec_con.append(False)
    else: 
        rec.append(False)
        prediction_validityA.append(False)
        prediction_validityB.append(False)
        rec_A.append(False)
        rec_B.append(False)
        rec_stoich.append(False)
        rec_con.append(False) """
        

if len(rec)>0: 
    rec_accuracy= sum(1 for entry in rec if entry) / len(rec)
    rec_accuracyA = sum(1 for entry in rec_A if entry) / len(rec_A)
    rec_accuracyB = sum(1 for entry in rec_B if entry) / len(rec_B)
    rec_accuracy_stoich = sum(1 for entry in rec_stoich if entry) / len(rec_stoich)
    rec_accuracy_con = sum(1 for entry in rec_con if entry) / len(rec_con)
else: 
    rec_accuracy = 0
    rec_accuracyA = 0
    rec_accuracyB = 0
    rec_accuracy_stoich = 0
    rec_accuracy_con = 0
validityA = sum(1 for entry in prediction_validityA if entry) / len(prediction_validityA)
validityB = sum(1 for entry in prediction_validityB if entry) / len(prediction_validityB)

print(dir_name+'reconstruction_metrics.txt')
with open(dir_name+'reconstruction_metrics.txt', 'w') as f:
    f.write("Full rec: %.4f %% Rec MonA: %.4f %% Rec MonB: %.4f %% Rec Stoichiometry: %.4f %% Rec Conectivity: %.4f %%  " % (100*rec_accuracy, 100*rec_accuracyA, 100*rec_accuracyB, 100*rec_accuracy_stoich, 100*rec_accuracy_con))
    f.write("Rec monomer A val: %.4f %% Rec monomer B val: %.4f %% "% (100*validityA, 100*validityB,))
    f.write(str(len(rec)))