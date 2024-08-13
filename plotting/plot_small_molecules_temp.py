from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import Draw
from math import ceil


def draw_molecule(s):
    mol = Chem.MolFromSmiles(s)

    img = Draw.MolToImage(mol,kekulize=False)

    return img


def draw_reactions(s):
    # Create a reaction object
    reaction = AllChem.ReactionFromSmarts(s)

    # Generate a 2D depiction of the reaction
    d2d = rdMolDraw2D.MolDraw2DCairo(400, 200)  # Set the size of the drawing
    d2d.DrawReaction(reaction)  # Draw the reaction
    d2d.FinishDrawing()
    img = d2d.GetDrawingText()

    # Display the image
    return img



def draw_molecule_grid(smiles_list, grid_size=(5,1), figsize=(10, 2), image_spacing=0.01, save_path=None, labels=None):
    num_molecules = len(smiles_list)
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for i, ax in enumerate(axes.flat):
        if i < num_molecules:
            try: 
                mol_img = draw_molecule(smiles_list[i])
            except: 
                pass
            ax.imshow(mol_img)
            ax.axis('off')

            # Add text overlay
            if labels is not None:
                ax.text(0.5, 0.5, labels[i], fontsize=15, color='black', alpha=0.3,
                        ha='center', va='center', transform=ax.transAxes, weight='bold')

    # Adjust spacing
    plt.subplots_adjust(left=image_spacing, right=1-image_spacing,
                        top=1-image_spacing, bottom=image_spacing,
                        wspace=0.05, hspace=image_spacing)
    
    

    # Save the plot as an image if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
smiles_list = [
        "CCO",  # Ethanol
    "C1=CC=CC=C1",  # Benzene
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "C(C1C(C(C(C(O1)O)O)O)O)O",  # Glucose
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "C",  # Methane
    "CC1(C)S[C@@H]2[C@H](N2C1=O)C(=O)O",  # Penicillin
    "C(Cl)(Cl)Cl",  # Chloroform
    "CN1CCCC1C2=CN=CC=C2",  # Nicotine
    "CC(=O)NC1=CC=C(O)C=C1",  # Paracetamol
    "COC1=C2C=CC3=C4C2=C(C=C1)C(=NCC3=N4)O",  # Quinine
    "C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O",  # Sucrose
    "CC1=CC(=C2C3=C(C=C(C=C3)O)C(C=C2C1)(C)C)C(C)C",  # Tetrahydrocannabinol (THC)
    "C(C(C(=O)CO)O)O",  # Vitamin C
    "CC1=CC=C(C=C1)C2=CC=CC(=C2)C(=O)O",  # Warfarin
    "C(C(C(C(C=O)O)O)O)O",  # Xylose
    "C(C1C(C(C(C(O1)O)O)O)O)OC2C(C(C(C(O2)CO)O)O)O",  # Lactose
    "CC(=CC(=O)NC1=CC=C(C=C1)O)C(C)CCCC(C)C",  # Capsaicin
    "COC1=CC(=CC=C1O)C=O"  # Vanillin
]
smiles_list = [
    "CC(C(=O)NCC(C(=O)O)N)CC",  # L-Theanine
    "C1=C2C=CC=CC2=CC=C1",  # Azulene
    "C1=CC=C2C(=O)OC=CC2=C1",  # Coumarin
    "COC1=C(C=CC(=C1)C=CC(=O)O)O",  # Ferulic Acid
    "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C2=O)C=CC(=C3)O)O)O)O"  # Quercetin
]

draw_molecule_grid(smiles_list, image_spacing=0.01, save_path="molecules_samplingVAE.png", labels=None)