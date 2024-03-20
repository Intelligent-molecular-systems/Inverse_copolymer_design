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
dir_name= os.path.join(main_dir_path,'Checkpoints/')

metrics_rec = {}
metrics_gen = {}
subdirs = [x[0] for x in os.walk(dir_name)]
all_results = []
for subdir in subdirs: 

    print(f'Validity check of validation set using inference decoding')

    try:
        with open(os.path.join(subdir,'all_val_prediction_strings.pkl'), 'rb') as f:
            all_predictions=pickle.load(f)
        with open(os.path.join(subdir,'all_val_real_strings.pkl'), 'rb') as f:
            all_real=pickle.load(f)
        # Remove all '_' from the strings (EOS token)
        all_predictions=[s.split('_', 1)[0] for s in all_predictions]
        all_real=[s.split('_', 1)[0] for s in all_real]


        with open(os.path.join(subdir,'all_val_prediction_strings.txt'), 'w') as f:
            for s in all_predictions:
                f.write(s+'\n')
        with open(os.path.join(subdir,'all_val_real_strings.txt'), 'w') as f:
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
                print("HI")
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
        print(len(rec))

        with open(os.path.join(subdir,'reconstruction_metrics.txt'), 'w') as f:
            f.write("Full rec: %.4f %% Rec MonA: %.4f %% Rec MonB: %.4f %% Rec Stoichiometry: %.4f %% Rec Conectivity: %.4f %%  " % (100*rec_accuracy, 100*rec_accuracyA, 100*rec_accuracyB, 100*rec_accuracy_stoich, 100*rec_accuracy_con))
            f.write("Rec monomer A val: %.4f %% Rec monomer B val: %.4f %% "% (100*validityA, 100*validityB,))
            f.write(str(len(rec)))
    except: 
        pass