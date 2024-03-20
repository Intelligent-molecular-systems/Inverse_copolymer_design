# Class for Smiles Enumeration and canonicalization,adapted from https://github.com/EBjerrum/SMILES-enumeration
# Extended for polymer strings including stoichiometry and connectivity
from rdkit import Chem
from rdkit.Chem import AllChem
from data_processing.rdkit_poly import make_monomer_mols, make_polymer_mol
import numpy as np
import re
from data_processing.data_utils import *
from data_processing.Function_Featurization_Own import poly_smiles_to_graph


class SmilesEnumCanon(object):
    """SMILES Enumerator and canonicalizer
    
    #Arguments
        enum: Enumerate the SMILES during transform
        renumber_poly_position: for polymer smiles only, renumber connection points
        stoich_con_info: if True the smiles contains also stoichiometry and connectivity information
    """
    def __init__(self, isomericSmiles=True):
        self.isomericSmiles = isomericSmiles

    def canonicalize(self, smiles, monomer_only=False, stoich_con_info=True):
        
        if monomer_only:
            try:
                mol = Chem.MolFromSmiles(smiles)
                return Chem.MolToSmiles(mol)
            except: 
                return 'invalid_monomer_string'
        else: 
            try:
                # Check if it is a valid polymer string
                mol = (make_polymer_mol(smiles.split("|")[0], 0, 0,  # smiles
                                        fragment_weights=smiles.split("|")[1:-1]),  # fraction of each fragment
                                        smiles.split("<")[1:]) # Connectivity
                if stoich_con_info: 
                    smiles_only = smiles.split("|")[0]
                    mol = Chem.MolFromSmiles(smiles_only)
                poly_smiles = self.renumber_polymerization_position(Chem.MolToSmiles(mol), smiles, stoich_con_info)
                return poly_smiles
            except:
                return 'invalid_polymer_string'

    def calculate_morgan_fingerprint(self, mol):
        
        m = Chem.MolFromSmiles(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        return fp

    def poly_smiles_to_molecule(self, poly_input):
        '''
        Turns polymer smiles string into mol object
        '''

        # Turn into RDKIT mol object
        mols = make_monomer_mols(poly_input.split("|")[0], 0, 0,  # smiles
                                fragment_weights=poly_input.split("|")[1:-1])
        
        return mols
        
    def randomize_smiles(self, smiles, nr_enum=1, renumber_poly_position=True, stoich_con_info=True):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        all_enumerated_smiles=[]
        for i in range(nr_enum):
            if stoich_con_info:
                smiles_only = smiles.split("|")[0]
            else:
                smiles_only=smiles
            m = Chem.MolFromSmiles(smiles_only)
            ans = list(range(m.GetNumAtoms()))
            np.random.shuffle(ans)
            nm = Chem.RenumberAtoms(m,ans)
            enumerated_smiles = Chem.MolToSmiles(nm, canonical=False, isomericSmiles=self.isomericSmiles)

            # renumber polymer position: renumber the smiles connection points, but adjust the stoichiometry if monomers are swapped in string 
            if renumber_poly_position: 
                all_enumerated_smiles.append(self.renumber_polymerization_position(enumerated_smiles, smiles, stoich_con_info))
            else: all_enumerated_smiles.append(enumerated_smiles)
        return all_enumerated_smiles
    
    def renumber_polymerization_position(self, input_string, original_smiles=None, stoich_con_info=False):
        # TODO: for asymmetric connectivity between monomers, the connectivity also needs to be changed if the monomers are swapped/enumerated (for now not needed)


        # Find all occurrences of the pattern [*:#]
        matches = re.findall(r'\[\*:(\d+)\]', input_string)
        # Determine which number to start at (in case it is only a B monomer starting with connection site 3 this should still be the case)
        min_connect_nr = min([int(match) for match in matches])

        # Create a dictionary to map old positions to new positions
        position_mapping = {old_pos: str(new_pos) for new_pos, old_pos in enumerate(matches, start=min_connect_nr)}

        # Replace occurrences in the input string
        output_string = re.sub(r'\[\*:(\d+)\]', lambda match: f'[*:{position_mapping[match.group(1)]}]', input_string)


        # If stoich con info is given, and monomers were swapped, stoichiometry should be changed 
        if stoich_con_info:
            try: monA, monB = original_smiles.split("|")[0].split(".")
            except: raise Exception("Specifiy stoich_con_info as False or provide a SMILES that contains stoichiometric information")
            # Regular expression pattern to match [*:#]
            pattern = r"\[\*:\d+\]"
            # Replacement string
            replacement = "[*]"
            # Use re.sub to replace the matched patterns
            monA_mod = self.canonicalize(re.sub(pattern, replacement, monA), monomer_only=True)
            monB_mod = self.canonicalize(re.sub(pattern, replacement, monB), monomer_only=True)

            try: 
                monA_en, monB_en = output_string.split("|")[0].split(".")
                stoich_con = "|".join(original_smiles.split("|")[1:])
            except: raise Exception("Specifiy stoich_con_info as False or provide a SMILES that contains stoichiometric information")
            monA_en_mod = self.canonicalize(re.sub(pattern, replacement, monA_en), monomer_only=True)
            monB_en_mod = self.canonicalize(re.sub(pattern, replacement, monB_en), monomer_only=True)

            #monomers not swapped: no action needed 
            if monA_mod==monA_en_mod and monB_mod==monB_en_mod:
                return output_string+"|"+stoich_con
            #swapped: swap also the stoichiometry
            elif monA_mod==monB_en_mod and monB_mod==monA_en_mod:
                stoich = "|".join(stoich_con.split("|")[:-1])
                con = stoich_con.split("|")[-1]
                # Regular expression pattern to capture the floating-point numbers
                pattern = r"(\d+\.\d+)\|(\d+\.\d+)"
                # Use re.sub to swap the two numbers
                stoich_swapped = re.sub(pattern, r"\2|\1", stoich)
                return output_string+"|"+stoich_swapped+"|"+con
            else: 
                raise Exception("Could not determine whether monomers were swapped! Please check for enumeration/canonicalization error")
        return output_string

    def transform(self, smiles):
        """Perform an enumeration (randomization) and vectorization of a Numpy array of smiles strings
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
        """
        one_hot =  np.zeros((smiles.shape[0], self.pad, self._charlen),dtype=np.int8)
        
        if self.leftpad:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j,c in enumerate(ss):
                    one_hot[i,j+diff,self._char_to_int[c]] = 1
            return one_hot
        else:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                for j,c in enumerate(ss):
                    one_hot[i,j,self._char_to_int[c]] = 1
            return one_hot

      
    def reverse_transform(self, vect):
        """ Performs a conversion of a vectorized SMILES to a smiles strings
        charset must be the same as used for vectorization.
        #Arguments
            vect: Numpy array of vectorized SMILES.
        """       
        smiles = []
        for v in vect:
            #mask v 
            v=v[v.sum(axis=1)==1]
            #Find one hot encoded index with argmax, translate to char and join to string
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)


if __name__ == "__main__":
    #Test enumeration
    sm_en = SmilesEnumCanon()
    rand_sms = []
    poly_input="[*:1]c1c(O)cc(O)c([*:2])c1O.[*:3]c1cc(F)c([*:4])cc1F|0.75|0.25|<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125"
    rand_sms = sm_en.randomize_smiles(poly_input, nr_enum=3)
    print(rand_sms)
    # Test canonicalization (should be only one element in set)
    can_sms=set(list(map(sm_en.canonicalize, rand_sms)))
    print(can_sms)

    
    


        
