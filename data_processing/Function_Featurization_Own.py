import data_processing.featurization as ft
from rdkit import Chem
from rdkit.Chem import Descriptors
from data_processing.rdkit_poly import make_polymer_mol, make_mol
from copy import deepcopy
from torch_geometric.data import Data
import torch

def poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan=None):
    '''
    Turns adjusted polymer smiles string into PyG data objects
    '''

    # Turn into RDKIT mol object
    mol = (make_polymer_mol(poly_input.split("|")[0], 0, 0,  # smiles
                            fragment_weights=poly_input.split("|")[1:-1]),  # fraction of each fragment
           poly_input.split("<")[1:])

    # Set some variables needed later
    n_atoms = 0  # number of atoms
    n_bonds = 0  # number of bonds
    f_atoms = []  # mapping from atom index to atom features
    f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
    w_bonds = []  # mapping from bond index to bond weight
    w_atoms = []  # mapping from atom index to atom weight
    a2b = []  # mapping from atom index to incoming bond indices
    b2a = []  # mapping from bond index to the index of the atom the bond is coming from
    b2revb = []  # mapping from bond index to the index of the reverse bond

    # ============
    # Polymer mode
    # ============
    m = mol[0]  # RDKit Mol object
    rules = mol[1]  # [str], list of rules
    # parse rules on monomer connections
    polymer_info, degree_of_polym = ft.parse_polymer_rules(
        rules)
    # make molecule editable
    rwmol = Chem.rdchem.RWMol(m)
    # tag (i) attachment atoms and (ii) atoms for which features needs to be computed
    # also get map of R groups to bonds types, e.f. r_bond_types[*1] -> SINGLEw
    rwmol, r_bond_types = ft.tag_atoms_in_repeating_unit(rwmol)

    # -----------------
    # Get atom features
    # -----------------
    # for all 'core' atoms, i.e. not R groups, as tagged before. Do this here so that atoms linked to
    # R groups have the correct saturation
    f_atoms = [ft.atom_features(
        atom) for atom in rwmol.GetAtoms() if atom.GetBoolProp('core') is True]
    w_atoms = [atom.GetDoubleProp(
        'w_frag') for atom in rwmol.GetAtoms() if atom.GetBoolProp('core') is True]

    n_atoms = len(f_atoms)

    # remove R groups -> now atoms in rdkit Mol object have the same order as self.f_atoms
    rwmol = ft.remove_wildcard_atoms(rwmol)

    # Initialize atom to bond mapping for each atom
    for _ in range(n_atoms):
        a2b.append([])
    rwmol

    # ---------------------------------------
    # Get bond features for separate monomers
    # ---------------------------------------

    # Here we do not add atom features like in polymer paper
    for a1 in range(n_atoms):
        for a2 in range(a1 + 1, n_atoms):
            bond = rwmol.GetBondBetweenAtoms(a1, a2)

            if bond is None:
                continue

            # get bond features
            f_bond = ft.bond_features(bond)

            # append bond features twice
            f_bonds.append(f_bond)
            f_bonds.append(f_bond)
            # Update index mappings
            b1 = n_bonds
            b2 = b1 + 1
            a2b[a2].append(b1)  # b1 = a1 --> a2
            b2a.append(a1)
            a2b[a1].append(b2)  # b2 = a2 --> a1
            b2a.append(a2)
            b2revb.append(b2)
            b2revb.append(b1)
            w_bonds.extend([1.0, 1.0])  # edge weights of 1.
            n_bonds += 2

    # ---------------------------------------------------
    # Get bond features for bonds between repeating units
    # ---------------------------------------------------
    # we duplicate the monomers present to allow (i) creating bonds that exist already within the same
    # molecule, and (ii) collect the correct bond features, e.g., for bonds that would otherwise be
    # considered in a ring when they are not, when e.g. creating a bond between 2 atoms in the same ring.
    rwmol_copy = deepcopy(rwmol)
    _ = [a.SetBoolProp('OrigMol', True) for a in rwmol.GetAtoms()]
    _ = [a.SetBoolProp('OrigMol', False)
         for a in rwmol_copy.GetAtoms()]
    # create an editable combined molecule
    cm = Chem.CombineMols(rwmol, rwmol_copy)
    cm = Chem.RWMol(cm)

    # for all possible bonds between monomers:
    # add bond -> compute bond features -> add to bond list -> remove bond
    for r1, r2, w_bond12, w_bond21 in polymer_info:

        # get index of attachment atoms
        a1 = None  # idx of atom 1 in rwmol
        a2 = None  # idx of atom 1 in rwmol --> to be used by MolGraph
        _a2 = None  # idx of atom 1 in cm --> to be used by RDKit
        for atom in cm.GetAtoms():
            # take a1 from a fragment in the original molecule object
            if f'*{r1}' in atom.GetProp('R') and atom.GetBoolProp('OrigMol') is True:
                a1 = atom.GetIdx()
            # take _a2 from a fragment in the copied molecule object, but a2 from the original
            if f'*{r2}' in atom.GetProp('R'):
                if atom.GetBoolProp('OrigMol') is True:
                    a2 = atom.GetIdx()
                elif atom.GetBoolProp('OrigMol') is False:
                    _a2 = atom.GetIdx()

        if a1 is None:
            print(poly_input)
            raise ValueError(f'cannot find atom attached to [*:{r1}]')
        if a2 is None or _a2 is None:
            print(poly_input)
            raise ValueError(f'cannot find atom attached to [*:{r2}]')

        # create bond
        order1 = r_bond_types[f'*{r1}']
        order2 = r_bond_types[f'*{r2}']
        if order1 != order2:
            raise ValueError(f'two atoms are trying to be bonded with different bond types: '
                             f'{order1} vs {order2}')
        cm.AddBond(a1, _a2, order=order1)
        Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL)

        # get bond object and features
        bond = cm.GetBondBetweenAtoms(a1, _a2)
        f_bond = ft.bond_features(bond)

        f_bonds.append(f_bond)
        f_bonds.append(f_bond)

        # Update index mappings
        b1 = n_bonds
        b2 = b1 + 1
        a2b[a2].append(b1)  # b1 = a1 --> a2
        b2a.append(a1)
        a2b[a1].append(b2)  # b2 = a2 --> a1
        b2a.append(a2)
        b2revb.append(b2)
        b2revb.append(b1)
        w_bonds.extend([w_bond12, w_bond21])  # add edge weights
        n_bonds += 2

        # remove the bond
        cm.RemoveBond(a1, _a2)
        Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL)

    # ------------------
    # Make ensemble molecular weight for self-supervised learning
    # ------------------

    monomer_smiles = poly_input.split("|")[0].split('.')
    # if smiles are canonicalized we want to also keep the non-canonical input
    if poly_input_nocan:
        monomer_smiles_nocan = poly_input_nocan.split("|")[0].split('.')
    else: monomer_smiles_nocan=None
        
    monomer_weights = poly_input.split("|")[1:-1]

    mol_mono_1 = make_mol(monomer_smiles[0], 0, 0)
    mol_mono_2 = make_mol(monomer_smiles[1], 0, 0)

    M_ensemble = float(monomer_weights[0]) * Descriptors.ExactMolWt(
        mol_mono_1) + float(monomer_weights[1]) * Descriptors.ExactMolWt(mol_mono_2)

    # -------------------------------------------
    # Make own pytroch geometric data object. Here we try follow outputs of above featurization: f_atoms, f_bonds, a2b, b2a
    # -------------------------------------------
    # PyG data object is: Data(x, edge_index, edge_attr, y, **kwargs)

    # create node feature matrix,
    X = torch.empty(n_atoms, len(f_atoms[0]))
    for i in range(n_atoms):
        X[i, :] = torch.FloatTensor(f_atoms[i])
    # associated atom weights we alread found
    W_atoms = torch.FloatTensor(w_atoms)

    # get edge_index and associated edge attribute and edge weight
    # edge index is of shape [2, num_edges],  edge_attribute of shape [num_edges, num_node_features], edge_weights = [num_edges]
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_attr = torch.empty(0, len(f_bonds[0]))
    W_bonds = torch.empty(0, dtype=torch.float)
    for i in range(n_atoms):
        # pick atom
        atom = torch.LongTensor([i])
        # find number of bonds to that atom. a2b is mapping from atom to bonds
        num_bonds = len(a2b[i])

        # create graph connectivivty for that atom
        atom_repeat = atom.repeat(1, num_bonds)
        # a2b is mapping from atom to incoming bonds, need b2a to map these bonds to atoms they originated from
        neigh_atoms = [b2a[bond] for bond in a2b[i]]
        edges = torch.LongTensor(neigh_atoms).reshape(1, num_bonds)
        edge_idx_atom = torch.cat((atom_repeat, edges), dim=0)
        # append connectivity of atom to edge_index
        edge_index = torch.cat((edge_index, edge_idx_atom), dim=1)

        # Find weight of bonds
        # weight of bonds attached to atom
        W_bond_atom = torch.FloatTensor([w_bonds[bond] for bond in a2b[i]])
        W_bonds = torch.cat((W_bonds, W_bond_atom), dim=0)

        # find edge attribute
        edge_attr_atom = torch.FloatTensor([f_bonds[bond] for bond in a2b[i]])
        edge_attr = torch.cat((edge_attr, edge_attr_atom), dim=0)

    # create PyG Data object
    graph = Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                 y1=poly_label1, y2=poly_label2, monomer_smiles=monomer_smiles, 
                 W_atoms=W_atoms, W_bonds=W_bonds, M_ensemble=M_ensemble, monomer_smiles_nocan=monomer_smiles_nocan)

    return graph

# %%
