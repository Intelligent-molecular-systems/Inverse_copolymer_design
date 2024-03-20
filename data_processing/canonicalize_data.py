import pickle
import pandas as pd
import re
import sys, os
from functools import partial
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from data_processing.Smiles_enum_canon import SmilesEnumCanon





df = pd.read_csv(main_dir_path+'/data/dataset-combined-poly_chemprop.csv')

all_poly_inputs = []
all_labels1 = []
all_labels2 = []
all_mono_combs = []
stoichiometry_connectivity_combs = []
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    poly_label1 = df.loc[i, 'EA vs SHE (eV)']
    poly_label2 = df.loc[i, 'IP vs SHE (eV)']
    all_labels1.append(poly_label1)
    all_labels2.append(poly_label2)
    all_poly_inputs.append(poly_input)
    all_mono_combs.append(poly_input.split("|")[0])

#canonicalization only
""" sm_can = SmilesEnumCanon()
new_polys = list(map(sm_can.canonicalize, all_poly_inputs))
new_entries = []
for i, canonical_poly_sm in enumerate(new_polys):
    new_entries.append(
        {
        'poly_chemprop_input': canonical_poly_sm,
        'EA vs SHE (eV)': all_labels1[i],
        'IP vs SHE (eV)': all_labels2[i],
        "poly_chemprop_input_nocan":df.loc[i, 'poly_chemprop_input']
        }
        )

df_new = pd.DataFrame(new_entries)
df_new.to_csv('dataset-combined-canonical-poly_chemprop.csv', index=False)
print(df_new.head(20)) """

#enumeration only
#nr_enumerations=1: data set size stays the same but the monomers are written in many different ways because they occur quite often in the dataset.
""" nr_enumerations = 1
sm_en = SmilesEnumCanon()
randomize_smiles_fixed_enums = partial(sm_en.randomize_smiles, nr_enum=nr_enumerations)
# Use map with the fixed function
new_polys = list(map(randomize_smiles_fixed_enums, all_poly_inputs)) # List of lists
new_entries=[]
for i, enumerated_smiles_list in enumerate(new_polys):
    # for each i (original datapoint, multiple enumerations that have the same labels and same nocan input)
    for enumerated_smiles in enumerated_smiles_list:
        new_entries.append(
            {
            'poly_chemprop_input': enumerated_smiles,
            'EA vs SHE (eV)': all_labels1[i],
            'IP vs SHE (eV)': all_labels2[i],
            "poly_chemprop_input_nocan":df.loc[i, 'poly_chemprop_input']
            }
            )

df_new = pd.DataFrame(new_entries)
df_new.to_csv('dataset-combined-enumerated-poly_chemprop.csv', index=False) """

#Different technique: Keep monomer order (don't mix A and B position) but use different enumerations for the monomers. 
# Keeps the A and B space kind of separated for the decoder
# currently only works with 1 enumeration
nr_enumerations = 1
sm_en = SmilesEnumCanon()
replacement_mon_comb={}
all_mono_combs_unique=list(set(all_mono_combs))
for c in all_mono_combs_unique:
    # split monomers and enumerate them once
    monA=c.split('.')[0]
    monB=c.split('.')[1]
    monA_en=sm_en.randomize_smiles(monA, nr_enum=1, renumber_poly_position=True, stoich_con_info=False)[0]
    monB_en=sm_en.randomize_smiles(monB,  nr_enum=1, renumber_poly_position=True, stoich_con_info=False)[0]
    replacement_mon_comb[c]='.'.join([monA_en,monB_en])
    
all_mono_combs_en = [replacement_mon_comb[item] for item in all_mono_combs]
print(all_mono_combs_en[0])
print(all_poly_inputs[0])

# use the new enumerated monomer combs to change poly inputs to new poly inputs (keep stoich and con)
new_polys=["|".join([all_mono_combs_en[i],item.split("|",1)[1]]) for i, item in enumerate(all_poly_inputs)]
print(new_polys[0])

new_entries=[]
for i,new_poly in enumerate(new_polys):
    new_entries.append(
        {
        'poly_chemprop_input': new_poly,
        'EA vs SHE (eV)': all_labels1[i],
        'IP vs SHE (eV)': all_labels2[i],
        "poly_chemprop_input_nocan":df.loc[i, 'poly_chemprop_input']
        }
        )

df_new = pd.DataFrame(new_entries)
df_new.to_csv(main_dir_path+'/data/dataset-combined-enumerated2-poly_chemprop.csv', index=False)
print(df_new.head(20))
print('Done')