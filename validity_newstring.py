import torch
from data_utils import *
from rdkit_poly import *

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import pickle
from statistics import mean
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

vocab = load_vocab(vocab_file='poly_smiles_vocab_'+augment+'_'+tokenization+'_gbigsmileslike.txt')

# Directory to save results
dir_name = os.path.join(os.getcwd(),'Checkpoints_new/Model_beta='+str(args.beta)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_gbigsmileslike/')
if not os.path.exists(dir_name):
   os.makedirs(dir_name)

print(f'Validity check of validation set using inference decoding')

with open(dir_name+'all_val_prediction_strings.pkl', 'rb') as f:
    all_predictions=pickle.load(f)
with open(dir_name+'all_val_real_strings.pkl', 'rb') as f:
    all_real=pickle.load(f)

# C
print(all_predictions[0])
prediction_mols = list(map(poly_smiles_to_molecule, all_predictions))
prediction_validityA.append([mon[0] is not None for mon in prediction_mols])
predBvalid = []
for mon in prediction_mols:
    try: 
        predBvalid.append(mon[1] is not None)
    except: 
        predBvalid.append(False)

prediction_validityB.append(predBvalid)
#reconstructed_SmilesA = list(map(Chem.MolToSmiles, [mon[0] for mon in prediction_mols]))
    #reconstructed_SmilesB = list(map(Chem.MolToSmiles, [mon[1] for mon in prediction_validity]))


# Evaluation of validation set reconstruction accuracy (inference)
monomer_smiles_true = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_real] 
monomer_smiles_predicted = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_predictions]
monA_pred = [mon[0] for mon in monomer_smiles_predicted]
monB_pred = []
for mon in monomer_smiles_predicted: 
    try: monB_pred.append(mon[1])
    except: monB_pred.append(" ")
monA_true = [mon[0] for mon in monomer_smiles_true]
monB_true = [mon[1] for mon in monomer_smiles_true]

monomer_weights_predicted = [poly_smiles.split("|")[1:-1] for poly_smiles in all_predictions]
monomer_weights_real = [poly_smiles.split("|")[1:-1] for poly_smiles in all_real]
monomer_con_predicted = [poly_smiles.split("|")[-1].split("_")[0] for poly_smiles in all_predictions]
monomer_con_real = [poly_smiles.split("|")[-1].split("_")[0] for poly_smiles in all_real]


i=0
for l1,l2 in zip([pred.split("_")[0] for pred in all_predictions],[real.split("_")[0] for real in all_real]): 
    if l1 == l2: 
        i+=1
rec_accuracy = (float(i)/float(len(all_predictions)))

i=0
wrong_monA = []
wrongmonA_tanimoto_similarity = []
for l1,l2 in zip(monA_pred,monA_true): 
    if l1 == l2: 
        i+=1
    else:
        wrong_monA.append((l1,l2))
        #fp1 = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(l1))  # You can adjust the radius
        #fp2 = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(l2))
        #wrongmonA_tanimoto_similarity.append(DataStructs.cDataStructs.TanimotoSimilarity(fp1, fp2))
rec_accuracyA = (float(i)/float(len(monA_pred)))
all_a_mons = list(set(monA_true))
#fp_allA = [FingerprintMols.FingerprintMol(Chem.MolFromSmiles(x)) for x in all_a_mons]
#tanimoto_allA = []
#for i, A in enumerate(fp_allA):
#    for j,A2 in enumerate(fp_allA):
#        if not i==j:
#            tanimoto_allA.append(DataStructs.cDataStructs.TanimotoSimilarity(A, A2))

#all_b_mons = list(set(monB_true))
#fp_allB = [FingerprintMols.FingerprintMol(Chem.MolFromSmiles(x)) for x in all_b_mons]
#tanimoto_allB = []
#for i, B in enumerate(fp_allB):
#    for j,B2 in enumerate(fp_allB):
#        if not i==j:
#            tanimoto_allB.append(DataStructs.cDataStructs.TanimotoSimilarity(B, B2))
i = 0
for l1,l2 in zip(monB_pred,monB_true): 
    if l1 == l2: 
        i+=1
rec_accuracyB = (float(i)/float(len(monB_pred)))
i = 0
for l1,l2 in zip(monomer_smiles_predicted, monomer_smiles_true): 
    if l1 == l2: 
        i+=1
rec_accuracymons = (float(i)/float(len(monomer_smiles_true)))

i = 0
weight_pred_wrong = {
    "0.75":0,
    "0.5":0,
    "0.25":0,
}
weight_pred_right = {
    "0.75":0,
    "0.5":0,
    "0.25":0,
}
for l1,l2 in zip(monomer_weights_predicted,monomer_weights_real): 
    if l1 == l2: 
        i+=1
        weight_pred_right[l2[0]]+=1
    else:
        weight_pred_wrong[l2[0]]+=1

rec_accuracy_stoich = (float(i)/float(len(monomer_weights_predicted)))

i = 0
predicted_classes={0:0,1:0,2:0}
classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
for l1,l2 in zip(monomer_con_predicted,monomer_con_real): 
    if l1 == l2: 
        i+=1
    else: 
        class_idx = [i for i, x in enumerate(classes_con) if x==l1][0]
        predicted_classes[class_idx]+=1
rec_accuracy_con = (float(i)/float(len(monomer_con_predicted)))
print(predicted_classes)
prediction_validityA= [num for elem in prediction_validityA for num in elem]
prediction_validityB = [num for elem in prediction_validityB for num in elem]
classes_stoich = ['0.5|0.5','0.25|0.75','0.75|0.25']
validityA = sum(prediction_validityA)/len(prediction_validityA)
validityB = sum(prediction_validityB)/len(prediction_validityB)
whole_valid = 0
for i,_ in enumerate(all_predictions):
    if prediction_validityA[i] and prediction_validityB[i] and monomer_con_predicted[i] in classes_con and '|'.join(monomer_weights_predicted[i]) in classes_stoich:
        whole_valid+=1
validity = whole_valid/len(all_predictions)
print(dir_name+'reconstruction_metrics.txt')
with open(dir_name+'reconstruction_metrics.txt', 'w') as f:
    f.write("Full rec: %.4f %% Rec both monomers: %.4f %% Rec MonA: %.4f %% Rec MonB: %.4f %% Rec Stoichiometry: %.4f %% Rec Conectivity: %.4f %%  " % (100*rec_accuracy, 100*rec_accuracymons, 100*rec_accuracyA, 100*rec_accuracyB, 100*rec_accuracy_stoich, 100*rec_accuracy_con))
    f.write("Rec monomer A val: %.4f %% Rec monomer B val: %.4f %% "% (100*validityA, 100*validityB,))
    f.write("Validity rec: %.4f %% "% (100*validity,))
