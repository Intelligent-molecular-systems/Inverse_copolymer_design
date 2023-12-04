import pandas as pd
df = pd.read_csv('dataset-poly_chemprop.csv')
all_monomer_combs = []
monomer_comb_property1_range = {}
monomer_comb_property2_range = {}
for i in range(len(df.loc[:, 'poly_chemprop_input'])):
    poly_input = df.loc[i, 'poly_chemprop_input']
    poly_label1 = df.loc[i, 'EA vs SHE (eV)']
    poly_label2 = df.loc[i, 'IP vs SHE (eV)']
    monomer_comb = "".join(poly_input.split("|")[0])
    stoich = "|".join(poly_input.split("|")[1:3])
    chain_arch = ''.join(poly_input.split("|")[-1].split(':')[1])
    if monomer_comb not in all_monomer_combs:
        all_monomer_combs.append(monomer_comb)
        monomer_comb_property1_range[monomer_comb] = {stoich+','+chain_arch:poly_label1}
        monomer_comb_property2_range[monomer_comb] = {stoich+','+chain_arch:poly_label2}

    else:
        monomer_comb_property1_range[monomer_comb][stoich+','+chain_arch]=poly_label1
        monomer_comb_property2_range[monomer_comb][stoich+','+chain_arch]=poly_label2
        
highesty1 = ''
max_diff = 0
for mon_comb, properties in monomer_comb_property1_range.items():
    l=list(properties.values())
    max_val = max(l)
    min_val = min(l)
    diff = abs(max_val-min_val)
    if diff > max_diff:
        highesty1 = mon_comb

print(highesty1)
print(monomer_comb_property1_range[highesty1])

highesty2 = ''
max_diff = 0
for mon_comb, properties in monomer_comb_property2_range.items():
    l=list(properties.values())
    max_val = max(l)
    min_val = min(l)
    diff = abs(max_val-min_val)
    if diff > max_diff:
        highesty2 = mon_comb

print(highesty2)
print(monomer_comb_property2_range[highesty2])

def monomer_combi(poly_smiles):

    monomer_comb = poly_input.split("|", 0)

    return monomer_comb
