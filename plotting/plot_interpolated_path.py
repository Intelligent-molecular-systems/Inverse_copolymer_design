from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import Draw
from math import ceil



# Function to create a placeholder image with Matplotlib
def create_placeholder_image_invalid(size=(200, 200), text="Invalid"):
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax.text(0.5, 0.5, text, fontsize=20, ha='center', va='center')
    ax.axis('off')
    fig.canvas.draw()
    
    # Convert the plot to an image
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image

def create_placeholder_image_white(size=(200, 200), text=""):
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
unique_mols = []
import os
scaling_factor=0.5
# path should match a text file generated with interpolate.py
file_path = os.path.join(os.getcwd(), 'plottinginterpolated_polymers_between_best_and_random.txt')
with open(file_path, 'r') as file:
    lines = file.readlines()
    print(lines)
    for l_idx, line in enumerate(lines):
        if l_idx<3:
            continue
        else:
            if not line in unique_mols:
                smiles = line.split("|")[0]
                print(line)
                con = line.split("|")[-1].split(':')[1]
                stoich = "|".join(line.split("|")[1:3])
                stoich_con_list.append("".join([labels_stoich[stoich],' ',labels_con[con]]))
                smiles_list.append(smiles)
                unique_mols.append(line)



# Convert new_molecules (list of SMILES strings) to images
import math
grid_size_x = math.ceil(len(unique_mols)/5)
grid_size_y = 5
placeholder_image_invalid = create_placeholder_image_invalid()
placeholder_image_white = create_placeholder_image_white()
if grid_size_x == 1:
    grid_size_x=2
    


molecule_images = [smiles_to_image(smiles) for smiles in smiles_list]
print(len(molecule_images))

# Create an empty grid
image_grid = np.empty((grid_size_x, grid_size_y), dtype=object)
inf_grid = np.empty((grid_size_x, grid_size_y), dtype=object)
inf_grid_obj = np.empty((grid_size_x, grid_size_y), dtype=object)



# Place molecule images in the grid
idx = 0
for i in range(grid_size_x):
    for j in range(grid_size_y):
        if idx<len(unique_mols):
            if molecule_images[idx]:
                image_grid[i, j] = molecule_images[idx]
                inf_grid[i,j] = stoich_con_list[idx]
            else: image_grid[i, j] = placeholder_image_invalid
            
        else: 
            image_grid[i, j] = placeholder_image_white
            inf_grid[i,j] = ""
        idx+=1
        
# molecule M1 should be at the center of the grid
#center_i, center_j = half_grid_size, half_grid_size
#image_grid[center_i, center_j] = smiles_to_image(smiles_list[len(smiles_list)//2])  # Assuming M1 is in the center of the list

# Plot the grid
fig, axes = plt.subplots(grid_size_x, grid_size_y, figsize=(10, 4))
plt.subplots_adjust(hspace=0.6)

# Plot each image in the grid
img_counter = 1
for i in range(grid_size_x):
    for j in range(grid_size_y):
        axes[i, j].imshow(image_grid[i, j])
        axes[i, j].axis('off')

        # Add text overlay
        if stoich_con_list is not None:
            axes[i, j].text(0.5, -0.25, inf_grid[i, j], ha='center', va='center', transform=axes[i, j].transAxes, fontsize=11, color='black', alpha=0.3, weight="bold")
        
        #Arrows to indicate interpolation
        arrow_start = (1.0, 0.5)  # Right center of the current subplot
        arrow_end = (1.3, 0.5)    # Left center of the next subplot

        # Add the arrow to the current axis pointing to the right
        if not img_counter>=len(unique_mols):    
            axes[i, j].annotate('', xy=arrow_end, xytext=arrow_start,
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=0.5, headwidth=4, headlength=6))
            print(img_counter, len(unique_mols))
        img_counter+=1



plt.savefig('interpolation_between_best_and_random.png', bbox_inches='tight', pad_inches=0.1, dpi=300)