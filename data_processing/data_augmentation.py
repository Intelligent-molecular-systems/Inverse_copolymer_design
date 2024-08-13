import os, sys
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

import numpy as np
import torch
from torch.utils.data import Dataset
from data_processing.Function_Featurization_Own import poly_smiles_to_graph
from data_processing.data_utils import *

import pickle
from statistics import mean
import pandas as pd
import re
import random

df = pd.read_csv(main_dir_path+'/data/dataset-poly_chemprop.csv')
monA_list = []
monB_list = []
stoichiometry_connectivity_combs = []
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    poly_label1 = df.loc[i, 'EA vs SHE (eV)']
    poly_label2 = df.loc[i, 'IP vs SHE (eV)']
    monA, monB = poly_input.split("|")[0].split(".")
    stoichiometry_connectivity_combs.append("|"+"|".join(poly_input.split("|")[1:]))
    monA_list.append(monA)
    monB_list.append(monB)
monAs = list(set(monA_list))
monBs = list(set(monB_list))
stoichiometry_connectivity_combs = list(set(stoichiometry_connectivity_combs))
print(len(monAs), len(monBs))

# Build a bigger dataset

# combine B monomers with n B monomers
n = 50

# create copy of B monomers and change the wildcards to be [*:1] and [*:2] 
monBs_mod = monBs.copy()
rep = {"[*:3]": "[*:1]", "[*:4]": "[*:2]"}
rep = dict((re.escape(k), v) for k, v in rep.items()) 
pattern = re.compile("|".join(rep.keys()))
for i, m in enumerate(monBs_mod): 
    monBs_mod[i] = pattern.sub(lambda x: rep[re.escape(x.group(0))], m)

# select randomly n B monomers to combine with B monomers
new_entries = []
for b1 in monBs_mod:
    for nn in range(n):
        b2 = random.choice(monB_list)
        new_mon_comb = ".".join([b1,b2])
        #stoich_con = random.choice(stoichiometry_connectivity_combs)
        for stoich_con in stoichiometry_connectivity_combs:
            new_poly = new_mon_comb + stoich_con
            new_entries.append(
            {
            'poly_chemprop_input': new_poly,
            'EA vs SHE (eV)': np.NAN,
            'IP vs SHE (eV)':  np.NAN
            }
            )
df_new = pd.DataFrame(new_entries)
df_new.to_csv(main_dir_path+'/data/dataset-augmented-poly_chemprop.csv', index=False)
df_combined = pd.concat([df,df_new], axis=0, ignore_index=True)
df_combined.to_csv(main_dir_path+'/data/dataset-combined-poly_chemprop.csv', index=False)
print(df_new.head(20))
print('Done')