# %% Packages
import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)
from data_processing.data_utils import *
from data_processing.rdkit_poly import *

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import pickle
from statistics import mean
import torch
import pandas as pd
import argparse
import numpy as np


def poly_smiles_to_molecule(poly_input):
    '''
    Turns adjusted polymer smiles string into PyG data objects
    '''

    # Turn into RDKIT mol object
    mols = make_monomer_mols(poly_input.split("|")[0], 0, 0,  # smiles
                            fragment_weights=poly_input.split("|")[1:-1])
    
    return mols

def valid_scores(smiles):
    return np.array(list(map(make_polymer_mol, smiles)), dtype=np.float32)


prediction_validityA = []
prediction_validityB = []


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

# Call data
parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original", default="original", choices=["augmented", "original", "augmented_canonical"])
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


dataset_type = "val" 
data_augment = "old" # new or old
dict_train_loader = torch.load('dataloaders_'+data_augment+'augment/dict_test_loader_'+augment+'_'+tokenization+'.pt')

vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')

# Directory to save results
dir_name= 'Checkpoints_new/Model_onlytorchseed_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

print(f'Validity check and metrics for newly generated samples')
std=0.025

with open(dir_name+'generated_polymers_from_seed_noise'+str(std)+'.pkl', 'rb') as f:
    all_predictions=pickle.load(f)

df = pd.read_csv('dataset-combined-poly_chemprop.csv')
all_polymers_data= []
all_train_polymers = []
dict_train_loader = torch.load('dataloaders_'+data_augment+'augment/dict_train_loader_'+augment+'_'+tokenization+'.pt')
for batch, graphs in enumerate(dict_train_loader):
    data = dict_train_loader[str(batch)][0]
    train_polymers_batch = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization).split('_')[0] for sample in range(len(data))]
    all_train_polymers.extend(train_polymers_batch)
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    all_polymers_data.append(poly_input)

 


# C
prediction_mols = list(map(poly_smiles_to_molecule, all_predictions))
for mon in prediction_mols: 
    try: prediction_validityA.append(mon[0] is not None)
    except: prediction_validityA.append(False)
    try: prediction_validityB.append(mon[1] is not None)
    except: prediction_validityB.append(False)


#predBvalid = []
#for mon in prediction_mols:
#    try: 
#        predBvalid.append(mon[1] is not None)
#    except: 
#        predBvalid.append(False)

#prediction_validityB.append(predBvalid)
#reconstructed_SmilesA = list(map(Chem.MolToSmiles, [mon[0] for mon in prediction_mols]))
#reconstructed_SmilesB = list(map(Chem.MolToSmiles, [mon[1] for mon in prediction_validity]))


# Evaluation of validation set reconstruction accuracy (inference)
monomer_smiles_predicted = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_predictions]
monA_pred = [mon[0] for mon in monomer_smiles_predicted]
monB_pred = []
for mon in monomer_smiles_predicted:
    try:
        monB_pred.append(mon[1])
    except:
        print(mon)
        monB_pred.append("")

monomer_smiles_train = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_train_polymers]
monA_t = [mon[0] for mon in monomer_smiles_train]
monB_t = [mon[1] for mon in monomer_smiles_train]

monomer_smiles_d = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_polymers_data]
monA_d = [mon[0] for mon in monomer_smiles_d]
monB_d = [mon[1] for mon in monomer_smiles_d]

monomer_weights_predicted = [poly_smiles.split("|")[1:-1] for poly_smiles in all_predictions]
monomer_con_predicted = [poly_smiles.split("|")[-1].split("_")[0] for poly_smiles in all_predictions]


#prediction_validityA= [num for elem in prediction_validityA for num in elem]
#prediction_validityB = [num for elem in prediction_validityB for num in elem]
validityA = sum(prediction_validityA)/len(prediction_validityA)
validityB = sum(prediction_validityB)/len(prediction_validityB)

# Novelty metrics
novel = 0
novel_pols=[]
for pol in all_predictions:
    if not pol[:-1] in all_train_polymers:
        novel+=1
        novel_pols.append(pol)
novelty = novel/len(all_predictions)
novel = 0
for pol in all_predictions:
    if not pol[:-1] in all_polymers_data:
        novel+=1
novelty_full_dataset = novel/len(all_predictions)
novelA = 0
for monA in monA_pred:
    if not monA in monA_d:
        novelA+=1
novelty_A = novelA/len(monA_pred)
novelB = 0
for monB in monB_pred:
    if not monB in monB_d:
        novelB+=1
novelty_B = novelB/len(monB_pred)

diversity = len(list(set(all_predictions)))/len(all_predictions)
diversity_novel = len(list(set(novel_pols)))/len(novel_pols)

classes_stoich = [['0.5','0.5'],['0.25','0.75'],['0.75','0.25']]
#if data_augment=='new':
#    classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
#else:
classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
whole_valid = 0
for i,_ in enumerate(all_predictions):
    if prediction_validityA[i] and prediction_validityB[i] and monomer_con_predicted[i] in classes_con and monomer_weights_predicted[i] in classes_stoich:
        whole_valid+=1
    else: 
        print(all_predictions[i])
validity = whole_valid/len(all_predictions)

# Some more statistics:
#variation monomer A, B, stoich, con
with open(dir_name+'seed_polymer.txt', 'r') as file:
    seed_string = file.read()
monomer_smiles_seed = seed_string.split("|")[0].split('.')
monA_s = monomer_smiles_seed[0]
monB_s = monomer_smiles_seed[1]

monomer_weights_seed = seed_string.split("|")[1:-1]
monomer_con_seed = seed_string.split("|")[-1].split("_")[0]

isn_a=0
isn_b=0
isn_s=0
isn_c=0
is_seed=0
for pol_ in all_predictions:
    pol = pol_.split('_')[0] 
    if pol == seed_string.split('_')[0]:
        is_seed+=1
for monA in monA_pred:
    if not monA==monA_s:
        isn_a+=1
for monB in monB_pred:
    if not monB == monB_s:
        isn_b+=1
for mw in monomer_weights_predicted:
    if not mw==monomer_weights_seed:
        isn_s+=1
for mc in monomer_con_predicted:
    if not mc == monomer_con_seed:
        isn_c+=1


with open(dir_name+'generated_polymers_from_seed_noise'+str(std)+'_metrics.txt', 'w') as f:
    f.write("Gen Mon A validity: %.4f %% Gen Mon B validity: %.4f %% "% (100*validityA, 100*validityB,))
    f.write("Gen validity: %.4f %% "% (100*validity,))
    f.write("Novelty: %.4f %% "% (100*novelty,))
    f.write("Novelty MonA full dataset: %.4f %% "% (100*novelty_A,))
    f.write("Novelty MonB full dataset: %.4f %% "% (100*novelty_B,))
    f.write("Novelty in full dataset: %.4f %% "% (100*novelty_full_dataset,))
    f.write("Diversity: %.4f %% "% (100*diversity,))
    f.write("Diversity (novel polymers): %.4f %% \n"% (100*diversity_novel,))
    f.write('x times seed: %s, variation in A: %s, B: %s, stoichiometry: %s, chain arch.: %s'%(str(is_seed),str(isn_a),str(isn_b),str(isn_s),str(isn_c)))

metrics = {
    'novelA':novelA,
    'novelB':novelB,
    'isn_a':isn_a,
    'isn_b':isn_b,
    'isn_c':isn_c,
    'isn_s':isn_s,
}
with open(dir_name+'generated_polymers_from_seed_noise'+str(std)+'_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)