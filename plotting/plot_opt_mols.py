from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import Draw
from math import ceil
import ast



# Function to create a placeholder image with Matplotlib
def create_placeholder_image(size=(200, 200), text="Invalid"):
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax.text(0.5, 0.5, text, fontsize=20, ha='center', va='center')
    ax.axis('off')
    fig.canvas.draw()
    
    # Convert the plot to an image
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image

# Function to convert SMILES to image or placeholder if invalid
def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol)
    else:
        return None
    


classes_stoich = [['0.5','0.5'],['0.25','0.75'],['0.75','0.25']]
#if data_augment=='new':
#    classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']
#else:
classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']

labels_stoich = {'0.5|0.5':'1:1','0.25|0.75':'1:3','0.75|0.25':'3:1'}
labels_con = {'0.5':'A','0.25':'R','0.375':'B'}

smiles_list = []
stoich_con_list = []
obj_val_list = []
IP_list = []
EA_list = []
poly_strings = []
import os
import re
scaling_factor=1.0
file_path = os.path.join(os.getcwd(), 'top20_mols_GA_correct_mimick_peak.txt')
with open(file_path, 'r') as file:
    lines = file.readlines()
    mols_dict=ast.literal_eval(lines[0])
    line2 = re.sub(r'tensor\(([^)]+)\)', r'\1', lines[1])
    props_dict=ast.literal_eval(line2)
    for iteration, poly_string in mols_dict.items():
        if not poly_string in poly_strings:
            smiles = poly_string.split("|")[0]
            con = poly_string.split("|")[-1].split(':')[1]
            stoich = "|".join(poly_string.split("|")[1:3])
            stoich_con_list.append("".join([labels_stoich[stoich],' ',labels_con[con]]))
            smiles_list.append(smiles)

            #properties
            print(props_dict[iteration])
            val_EA = float(props_dict[iteration][0])
            val_IP = float(props_dict[iteration][1])
            IP_list.append(val_IP)
            EA_list.append(val_EA)
            # val_IP = float(props_dict[iteration][2])
            #val_obj = (val_EA + abs(val_IP - 1.0))
            val_obj = (abs(val_EA + 2.64) + abs(val_IP - 1.61))
            obj_val_list.append(round(val_obj,3))

            poly_strings.append(poly_string)



# Convert new_molecules (list of SMILES strings) to images
grid_size_x = 2
grid_size_y = 5
placeholder_image = create_placeholder_image()

molecule_images = [smiles_to_image(smiles) for smiles in smiles_list[:10]]
print(len(molecule_images))

# Create an empty grid
image_grid = np.empty((grid_size_x, grid_size_y), dtype=object)
inf_grid = np.empty((grid_size_x, grid_size_y), dtype=object)
inf_grid_obj = np.empty((grid_size_x, grid_size_y), dtype=object)



# Place molecule images in the grid
idx = 0
for i in range(grid_size_x):
    for j in range(grid_size_y):
        if molecule_images[idx]:
            image_grid[i, j] = molecule_images[idx]
            inf_grid[i,j] = stoich_con_list[idx]
            inf_grid_obj[i,j] ="f(z)="+"{:.3f}".format(obj_val_list[idx])+"\n"+ "EA (eV): "+"{:.3f}".format(EA_list[idx])+"\n"+"IP (eV): "+"{:.3f}".format(IP_list[idx])
        else: image_grid[i, j] = placeholder_image
        idx += 1

# molecule M1 should be at the center of the grid
#center_i, center_j = half_grid_size, half_grid_size
#image_grid[center_i, center_j] = smiles_to_image(smiles_list[len(smiles_list)//2])  # Assuming M1 is in the center of the list

# Plot the grid
plt.rcParams.update({'font.size': 18, 'name':'Droid Sans'})  # Apply to all text elements

fig, axes = plt.subplots(grid_size_x, grid_size_y, figsize=(10, 4))
plt.subplots_adjust(hspace=0.6)

# Plot each image in the grid
for i in range(grid_size_x):
    for j in range(grid_size_y):
        axes[i, j].imshow(image_grid[i, j])
        axes[i, j].axis('off')

        # Add text overlay
        if stoich_con_list is not None:
            axes[i, j].text(0.5, 0.5, inf_grid[i, j], ha='center', va='center', transform=axes[i, j].transAxes, fontsize=11, color='black', alpha=0.3, weight="bold")
            axes[i, j].text(0.5, -0.25, inf_grid_obj[i, j], ha='center', va='center', transform=axes[i, j].transAxes, fontsize=11, color='black')




plt.savefig('optimal_mols_GA_correct_mimick_peak.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
import json
print(json.dumps(mols_dict))
print(obj_val_list)
print(np.mean(obj_val_list))

