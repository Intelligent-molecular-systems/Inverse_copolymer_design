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
import pandas as pd
import argparse


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
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original, augmented_canonical", default="augmented", choices=["augmented", "original", "augmented_canonical", "augmented_enum", "augmented_old"])
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



dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')
dataset_type = "test"
data_augment ="old"
vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
vocab = load_vocab(vocab_file=vocab_file)

# Directory to save results
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
dir_name = os.path.join(main_dir_path,'Checkpoints/', model_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


print(f'Validity check and metrics for newly generated samples')

with open(dir_name+'generated_polymers.pkl', 'rb') as f:
    all_predictions=pickle.load(f)

if augment=="augmented":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-poly_chemprop_v2.csv')
if augment=="augmented_old":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-poly_chemprop.csv')
elif augment=="augmented_canonical":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-canonical-poly_chemprop.csv')
elif augment=="augmented_enum":
    df = pd.read_csv(main_dir_path+'/data/dataset-combined-enumerated2_poly_chemprop.csv')
all_polymers_data= []
all_train_polymers = []

for batch, graphs in enumerate(dict_train_loader):
    data = dict_train_loader[str(batch)][0]
    train_polymers_batch = [combine_tokens(tokenids_to_vocab(data.tgt_token_ids[sample], vocab), tokenization=tokenization).split('_')[0] for sample in range(len(data))]
    all_train_polymers.extend(train_polymers_batch)
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    all_polymers_data.append(poly_input)

 

sm_can = SmilesEnumCanon()
all_predictions_can = list(map(sm_can.canonicalize, all_predictions))
all_train_can = list(map(sm_can.canonicalize, all_train_polymers))
all_pols_data_can = list(map(sm_can.canonicalize, all_polymers_data))


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
monomer_smiles_predicted = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
monA_pred = [mon[0] for mon in monomer_smiles_predicted]
monB_pred = [mon[1] for mon in monomer_smiles_predicted]

monomer_smiles_train = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_train_can]
monA_t = [mon[0] for mon in monomer_smiles_train]
monB_t = [mon[1] for mon in monomer_smiles_train]

monomer_smiles_d = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_pols_data_can]
monA_d = [mon[0] for mon in monomer_smiles_d]
monB_d = [mon[1] for mon in monomer_smiles_d]
unique_mons = list(set(monA_d) | set(monB_d))


monomer_weights_predicted = [poly_smiles.split("|")[1:-1] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']
monomer_con_predicted = [poly_smiles.split("|")[-1].split("_")[0] for poly_smiles in all_predictions_can if poly_smiles != 'invalid_polymer_string']


#prediction_validityA= [num for elem in prediction_validityA for num in elem]
#prediction_validityB = [num for elem in prediction_validityB for num in elem]
validityA = sum(prediction_validityA)/len(prediction_validityA)
validityB = sum(prediction_validityB)/len(prediction_validityB)

# Novelty metrics
novel = 0
novel_pols=[]
for pol in all_predictions_can:
    if not pol in all_train_can:
        novel+=1
        novel_pols.append(pol)
novelty = novel/len(all_predictions)
novel = 0
for pol in all_predictions_can:
    if not pol in all_pols_data_can:
        novel+=1
novelty_full_dataset = novel/len(all_predictions)
novelA = 0
for monA in monA_pred:
    if not monA in unique_mons:
        novelA+=1
novelty_A = novelA/len(monA_pred)
novelB = 0
for monB in monB_pred:
    if not monB in unique_mons:
        novelB+=1
novelty_B = novelB/len(monB_pred)

diversity = len(list(set(all_predictions)))/len(all_predictions)
diversity_novel = len(list(set(novel_pols)))/len(novel_pols)

classes_stoich = [['0.5','0.5'],['0.25','0.75'],['0.75','0.25']]
#if data_augment=='new':
#    classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
#else:
classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.125:0.125<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
whole_valid = len(monomer_smiles_predicted)
validity = whole_valid/len(all_predictions)
with open(dir_name+'generated_polymers.txt', 'w') as f:
    f.write("Gen Mon A validity: %.4f %% Gen Mon B validity: %.4f %% "% (100*validityA, 100*validityB,))
    f.write("Gen validity: %.4f %% "% (100*validity,))
    f.write("Novelty: %.4f %% "% (100*novelty,))
    f.write("Novelty MonA full dataset: %.4f %% "% (100*novelty_A,))
    f.write("Novelty MonB full dataset: %.4f %% "% (100*novelty_B,))
    f.write("Novelty in full dataset: %.4f %% "% (100*novelty_full_dataset,))
    f.write("Diversity: %.4f %% "% (100*diversity,))
    f.write("Diversity (novel polymers): %.4f %% "% (100*diversity_novel,))

with open(dir_name+'generated_polymers_examples.txt', 'w') as f:
    for e in all_predictions:
        f.write(f"{e}\n")