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




dir_name= os.path.join(main_dir_path,'Checkpoints/')
augment='augmented'
tokenization = "RT_tokenized"
dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')
vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
vocab = load_vocab(vocab_file=vocab_file)

metrics_rec = {}
metrics_gen = {}
subdirs = [x[0] for x in os.walk(dir_name)]
all_results = []
subdirs.pop(0)
if augment=="augmented":
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
all_train_can = list(map(sm_can.canonicalize, all_train_polymers))
all_pols_data_can = list(map(sm_can.canonicalize, all_polymers_data))

monomer_smiles_train = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_train_can]
monA_t = [mon[0] for mon in monomer_smiles_train]
monB_t = [mon[1] for mon in monomer_smiles_train]

monomer_smiles_d = [poly_smiles.split("|")[0].split('.') for poly_smiles in all_pols_data_can]
monA_d = [mon[0] for mon in monomer_smiles_d]
monB_d = [mon[1] for mon in monomer_smiles_d]



for subdir in subdirs: 
    try:
        prediction_validityA = []
        prediction_validityB = []

        print(f'Validity check and metrics for newly generated samples')

        with open(os.path.join(subdir,'generated_polymers.pkl'), 'rb') as f:
            all_predictions=pickle.load(f)
    
        all_predictions_can = list(map(sm_can.canonicalize, all_predictions))

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
        for pol in all_predictions:
            if not pol in all_pols_data_can:
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
        classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.125:0.125<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
        whole_valid = len(monomer_smiles_predicted)
        validity = whole_valid/len(all_predictions)
        with open(os.path.join(subdir,'generated_polymers.txt'), 'w') as f:
            f.write("Gen Mon A validity: %.4f %% Gen Mon B validity: %.4f %% "% (100*validityA, 100*validityB,))
            f.write("Gen validity: %.4f %% "% (100*validity,))
            f.write("Novelty: %.4f %% "% (100*novelty,))
            f.write("Novelty MonA full dataset: %.4f %% "% (100*novelty_A,))
            f.write("Novelty MonB full dataset: %.4f %% "% (100*novelty_B,))
            f.write("Novelty in full dataset: %.4f %% "% (100*novelty_full_dataset,))
            f.write("Diversity: %.4f %% "% (100*diversity,))
            f.write("Diversity (novel polymers): %.4f %% "% (100*diversity_novel,))

        with open(os.path.join(subdir,'generated_polymers_examples.txt'), 'w') as f:
            for e in all_predictions:
                f.write(f"{e}\n")
        print("worked")
    except: pass