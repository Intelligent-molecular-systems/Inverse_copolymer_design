import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.core.termination import Termination
from pymoo.core.population import Population
import argparse
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from scipy.spatial import distance
import pickle
import time


import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

from model.G2S_clean import *
from data_processing.data_utils import *
from data_processing.Function_Featurization_Own import poly_smiles_to_graph
from data_processing.rdkit_poly import make_polymer_mol


# setting device on GPU if available, else CPU
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original", default="augmented", choices=["augmented", "original"])
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="oldtok", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32)
parser.add_argument("--beta", default=1, help="option: <any number>, schedule", choices=["normalVAE","schedule"])
parser.add_argument("--loss", default="ce", choices=["ce","wce"])
parser.add_argument("--AE_Warmup", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initialization", default="random", choices=["random"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=0)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.1)
parser.add_argument("--max_alpha", type=float, default=0.1)
parser.add_argument("--epsilon", type=float, default=1)


args = parser.parse_args()

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "train" 
data_augment = "old" # new or old
dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')

num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

# Load model
# Create an instance of the G2S model from checkpoint
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
filepath = os.path.join(main_dir_path,'Checkpoints/', model_name,"model_best_loss.pt")
if os.path.isfile(filepath):
    if args.ppguided:
        model_type = G2S_VAE_PPguided
    else: 
        model_type = G2S_VAE
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model_config = checkpoint["model_config"]
    batch_size = model_config['batch_size']
    hidden_dimension = model_config['hidden_dimension']
    embedding_dimension = model_config['embedding_dim']
    model_config["max_alpha"]=args.max_alpha
    vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    vocab = load_vocab(vocab_file=vocab_file)
    if model_config['loss']=="wce":
        class_weights = token_weights(vocab_file)
        class_weights = torch.FloatTensor(class_weights)
        model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, loss_weights=class_weights, add_latent=add_latent)
    else: model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dimension,device,model_config,vocab,seed, add_latent=add_latent)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

dir_name = os.path.join(main_dir_path,'Checkpoints/', model_name)

class Property_optimization_problem(Problem):
    def __init__(self, model, x_min, x_max, objective_type):
        super().__init__(n_var=len(x_min), n_obj=2, n_constr=0, xl=x_min, xu=x_max)
        self.model_predictor = model
        self.weight_electron_affinity = 1  # Adjust the weight for electron affinity
        self.weight_ionization_potential = 5  # Adjust the weight for ionization potential
        self.weight_z_distances = 5  # Adjust the weight for distance between GA chosen z and reencoded z
        self.penalty_value = 100  # Adjust the weight for penalty of validity
        self.modified_solution = None # Initialize the class variable that later stores the recalculated latents
        self.modified_solution_history = []  # Initialize list to store modified solutions
        self.results_custom = {}
        self.eval_calls = 0
        self.objective_type = objective_type


    def _evaluate(self, x, out, *args, **kwargs):
        # Assuming x is a 1D array containing the 32 numerical parameters

        # Inference: forward pass NN prediciton of properties and beam search decoding from latent
        self.eval_calls += 1
        x_torch = torch.from_numpy(x).to(device).to(torch.float32) 
        print("Evaluation should be repaired")
        print(x)
        with torch.no_grad():
            predictions, _, _, _, y = self.model_predictor.inference(data=x_torch, device=device, sample=False, log_var=None)
        # Validity check of the decoded molecule + penalize invalid molecules
        prediction_strings, validity = self._calc_validity(predictions)
        
        invalid_mask = (validity == 0)
        zero_vector = np.zeros(self.n_var)
        validity_mask = np.all(x != zero_vector, axis=1)
        print(len(invalid_mask))
        print(validity_mask.shape[0])
        print(x.shape[0])
        print(np.array(y.cpu()).shape[0])
        print(out["F"])
        out["F"] = np.zeros((x.shape[0], 2))
        print(out["F"].shape[0])
        if self.objective_type=='mimick_peak':
            out["F"][validity_mask, 0] = self.weight_electron_affinity *  np.abs(np.array(y.cpu())[validity_mask,0]+2)
            out["F"][validity_mask, 1] = self.weight_ionization_potential * np.abs(np.array(y.cpu())[validity_mask,1] - 1.2)  # Bring the second property (ionization potential) as close to 1 as possible
        elif self.objective_type=='mimick_best':
            out["F"][validity_mask, 0] = self.weight_electron_affinity *  np.abs(np.array(y.cpu())[validity_mask,0]+2.64)
            out["F"][validity_mask, 1] = self.weight_ionization_potential * np.abs(np.array(y.cpu())[validity_mask,1] - 1.61)
        elif self.objective_type=='EAmin':
            out["F"][validity_mask, 0] = self.weight_electron_affinity *  np.array(y.cpu())[validity_mask,0]  # Minimize the first property (electron affinity)
            out["F"][validity_mask, 1] = self.weight_ionization_potential * np.abs(np.array(y.cpu())[validity_mask,1] - 1.0)  # Bring the second property (ionization potential) as close to 1 as possible
        elif self.objective_type =='max_gap':
            out["F"][validity_mask, 0] = self.weight_electron_affinity *  np.array(y.cpu())[validity_mask,0]  # Minimize the first property (electron affinity)
            out["F"][validity_mask, 1] = -self.weight_ionization_potential * np.array(y.cpu())[validity_mask,1]  # Maximize IP (is in general positive)
        out["F"][~validity_mask] += self.penalty_value


        
        # Encode and predict the valid molecules
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        y_p_after_encoding_valid, z_p_after_encoding_valid, all_reconstructions_valid, _=self._encode_and_predict_molecules(predictions_valid)
        expanded_y_p = np.array([y_p_after_encoding_valid.pop(0) if val == 1 else [np.nan,np.nan] for val in list(validity)])
        expanded_z_p = np.array([z_p_after_encoding_valid.pop(0) if val == 1 else [0] * 32 for val in list(validity)])
        all_reconstructions = [all_reconstructions_valid.pop(0) if val == 1 else "" for val in list(validity)]
        print("evaluation should not change")
        print(expanded_z_p)


        out["F_corrected"] = np.zeros((x.shape[0], 2))
        if self.objective_type=='mimick_peak':
            #out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]  # Minimize the first property (electron affinity)
            out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2) # Bring the first property (electron affinity) close to -2
            out["F_corrected"][~invalid_mask, 1] = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.2)  # Bring the second property (ionization potential) as close to 1 as possible
        elif self.objective_type=='mimick_best':
            #out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0]  # Minimize the first property (electron affinity)
            out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * np.abs(expanded_y_p[~invalid_mask, 0] + 2.64) # Bring the first property (electron affinity) close to -2
            out["F_corrected"][~invalid_mask, 1] = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.61)  # Bring the second property (ionization potential) as close to 1 as possible
        elif self.objective_type=='EAmin':
            out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0] # Bring the first property (electron affinity) close to -2
            out["F_corrected"][~invalid_mask, 1] = self.weight_ionization_potential * np.abs(expanded_y_p[~invalid_mask, 1] - 1.0) 
        elif self.objective_type =='max_gap':
            out["F_corrected"][~invalid_mask, 0] = self.weight_electron_affinity * expanded_y_p[~invalid_mask, 0] # Bring the first property (electron affinity) close to -2
            out["F_corrected"][~invalid_mask, 1] = -self.weight_ionization_potential * expanded_y_p[~invalid_mask, 1] 


        # results
        #print(out["F"])
        aggr_obj = np.sum(out["F"], axis=1)
        aggr_obj_corrected = np.sum(out["F_corrected"], axis=1)
        results_dict = {
            "objective":aggr_obj,
            "objective_corrected": aggr_obj_corrected,
            "latents_reencoded": x, 
            "predictions": y,
            "predictions_doublecorrect": expanded_y_p,
            "string_decoded": prediction_strings, 
        }
        self.results_custom[str(self.eval_calls)] = results_dict
    
    def _calc_validity(self, predictions):
        # Molecule validity check     
        # Return a boolean array indicating whether each solution is valid or not
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        mols_valid= []
        for _s in prediction_strings:
            poly_input = _s[:-1] # Last element is the _ char
            poly_input_nocan=None
            poly_label1 = np.nan
            poly_label2 = np.nan
            try: 
                poly_graph=poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan)
                mols_valid.append(1)
            except:
                poly_graph = None
                mols_valid.append(0)
        mols_valid = np.array(mols_valid) # List of lists
        return prediction_strings, mols_valid
    
    def _make_polymer_mol(self,poly_input):
        # If making the mol works, the string is considered valid
        try: 
            _ = (make_polymer_mol(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1]), poly_input.split("<")[1:])
            return 1
        # If not, it is considered invalid
        except: 
            return 0
    
    def _encode_and_predict_molecules(self, predictions):
        # create data that can be encoded again
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        data_list = []
        for i, s in enumerate(prediction_strings):
            poly_input = s[:-1] # Last element is the _ char
            poly_input_nocan=None
            poly_label1 = np.nan
            poly_label2 = np.nan
            g = poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan)
            if tokenization=="oldtok":
                target_tokens = tokenize_poly_input(poly_input=poly_input)
            elif tokenization=="RT_tokenized":
                target_tokens = tokenize_poly_input_RTlike(poly_input=poly_input)
            tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=target_tokens, vocab=vocab)
            g.tgt_token_ids = tgt_token_ids
            g.tgt_token_lens = tgt_lens
            g.to(device)
            data_list.append(g)
        data_loader = DataLoader(dataset=data_list, batch_size=64, shuffle=False)
        dict_data_loader = MP_Matrix_Creator(data_loader, device)

        #Encode and predict
        batches = list(range(len(dict_data_loader)))
        y_p = []
        z_p = []
        all_reconstructions = []
        with torch.no_grad():
            for i, batch in enumerate(batches):
                data = dict_data_loader[str(batch)][0]
                data.to(device)
                dest_is_origin_matrix = dict_data_loader[str(batch)][1]
                dest_is_origin_matrix.to(device)
                inc_edges_to_atom_matrix = dict_data_loader[str(batch)][2]
                inc_edges_to_atom_matrix.to(device)

                # Perform a single forward pass.
                reconstruction, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
                y_p.append(y.cpu().numpy())
                z_p.append(z.cpu().numpy())
                reconstruction_strings = [combine_tokens(tokenids_to_vocab(reconstruction[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(reconstruction))]
                all_reconstructions.extend(reconstruction_strings)
        #Return the predictions from the encoded latents
        y_p_flat = [sublist.tolist() for array_ in y_p for sublist in array_]
        z_p_flat = [sublist.tolist() for array_ in z_p for sublist in array_]
        self.modified_solution = z_p_flat

        return y_p_flat, z_p_flat, all_reconstructions, dict_data_loader

# Define a convergence termination class
class ConvergenceTermination(Termination):
    def __init__(self, conv_threshold, conv_generations, n_max_gen):
        super().__init__()
        self.conv_threshold = conv_threshold
        self.conv_generations = conv_generations
        self.n_max_gen = n_max_gen
        self.best_fit_values = []
        self.conv_counter = 0
        self.converged_solution_X = None
        self.converged_solution_F = None

    # check for the convergence criterion
    def _do_continue(self, algorithm):
        return self.perc < 1.0
    
    # check the convergence progress
    def _update(self, algorithm):
        best_fit = algorithm.pop.get("F").min()
        self.best_fit_values.append(best_fit)
        
        if algorithm.n_gen >= self.n_max_gen:
            return 1.0

        if len(self.best_fit_values) > self.conv_generations:
            conv_rate = abs(self.best_fit_values[-1] - self.best_fit_values[-self.conv_generations]) / self.conv_generations
            if conv_rate < self.conv_threshold:
                self.conv_counter += 1
                if self.conv_counter >= 5:
                    # store the termination object and use it to print the converged solution and the objective value later
                    self.converged_solution_X = algorithm.pop[np.argmin(algorithm.pop.get("F"))].get("X")
                    self.converged_solution_F = algorithm.pop.get("F")
                    print(f"Algorithm has converged after {algorithm.n_gen} generations.")
                    return 1.0
            else:
                self.conv_counter = 0
        return algorithm.n_gen / self.n_max_gen



# Determine the boundaries for the latent dimensions from training dataset
with open(dir_name+'latent_space_'+dataset_type+'.npy', 'rb') as f:
    latent_space = np.load(f)
min_values = np.amin(latent_space, axis=0).tolist()
max_values = np.amax(latent_space, axis=0).tolist()

cutoff=0.0

if not cutoff==0.0:
    transformed_min_values = []
    transformed_max_values = []
    for min_val, max_val in zip(min_values, max_values):
        #bounds are larger than in training set if cutoff value is negative
        # Calculate amount to cut off from each end (cutoff*100 %)
        cutoff_amount = cutoff * abs(max_val - min_val)
        # Adjust min and max values
        transformed_min = min_val + cutoff_amount
        transformed_max = max_val - cutoff_amount
        transformed_min_values.append(transformed_min)
        transformed_max_values.append(transformed_max)
    min_values = transformed_min_values
    max_values = transformed_max_values

    

# Initialize the problem
# options: max_gap, EAmin, mimick_peak, mimick_best
objective_type='EAmin'
problem = Property_optimization_problem(model, min_values, max_values, objective_type)

# Termination criterium
termination = ConvergenceTermination(conv_threshold=0.0025, conv_generations=20, n_max_gen=500)


# Define NSGA2 algorithm parameters
pop_size = 200
sampling = LatinHypercubeSampling()
crossover = SimulatedBinaryCrossover(prob=0.90, eta=20)
#crossover = SimulatedBinaryCrossover()
mutation = PolynomialMutation(prob=1.0 / problem.n_var, eta=30)

# Initialize the NSGA2 algorithm
# algorithm = MyCustomNSGA2(pop_size=pop_size,
#                   sampling=sampling,
#                   crossover=crossover,
#                   mutation=mutation,
#                   eliminate_duplicates=True)


from pymoo.core.repair import Repair
class correctSamplesRepair(Repair):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model_predictor = model

    def _do(self, problem, Z, **kwargs):
        # repair sampled points whole batch 
        #pop_Z = []
        #for i in range(len(Z)): 
        #    pop_Z.append(Z[i])
        # Inference: forward pass NN prediciton of properties and beam search decoding from latent
        # Repair the sampled population

        #pop_Z_np=np.array(pop_Z)
        #pop_Z_torch = torch.from_numpy(pop_Z_np).to(device).to(torch.float32)
        pop_Z_torch = torch.from_numpy(Z).to(device).to(torch.float32)
        with torch.no_grad():
            predictions, _, _, _, y = self.model_predictor.inference(data=pop_Z_torch, device=device, sample=False, log_var=None)
        # Validity check of the decoded molecule + penalize invalid molecules
        prediction_strings, validity = self._calc_validity(predictions)
        invalid_mask = (validity == 0)
        # Encode and predict the valid molecules
        predictions_valid = [j for j, valid in zip(predictions, validity) if valid]
        y_p_after_encoding_valid, z_p_after_encoding_valid, all_reconstructions_valid, _=self._encode_and_predict_molecules(predictions_valid)
        expanded_z_p = np.array([z_p_after_encoding_valid.pop(0) if val == 1 else [0] * 32 for val in list(validity)])

        print("repaired population")
        print(expanded_z_p)
        Z = Population().create(*expanded_z_p)
        return Z


    def _calc_validity(self, predictions):
        # Molecule validity check     
        # Return a boolean array indicating whether each solution is valid or not
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        mols_valid= []
        for _s in prediction_strings:
            poly_input = _s[:-1] # Last element is the _ char
            poly_input_nocan=None
            poly_label1 = np.nan
            poly_label2 = np.nan
            try: 
                poly_graph=poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan)
                mols_valid.append(1)
            except:
                poly_graph = None
                mols_valid.append(0)
        mols_valid = np.array(mols_valid) # List of lists
        return prediction_strings, mols_valid

    def _make_polymer_mol(self,poly_input):
        # If making the mol works, the string is considered valid
        try: 
            _ = (make_polymer_mol(poly_input.split("|")[0], 0, 0, fragment_weights=poly_input.split("|")[1:-1]), poly_input.split("<")[1:])
            return 1
        # If not, it is considered invalid
        except: 
            return 0
    
    def _encode_and_predict_molecules(self, predictions):
        # create data that can be encoded again
        prediction_strings = [combine_tokens(tokenids_to_vocab(predictions[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(predictions))]
        data_list = []
        for i, s in enumerate(prediction_strings):
            poly_input = s[:-1] # Last element is the _ char
            poly_input_nocan=None
            poly_label1 = np.nan
            poly_label2 = np.nan
            g = poly_smiles_to_graph(poly_input, poly_label1, poly_label2, poly_input_nocan)
            if tokenization=="oldtok":
                target_tokens = tokenize_poly_input(poly_input=poly_input)
            elif tokenization=="RT_tokenized":
                target_tokens = tokenize_poly_input_RTlike(poly_input=poly_input)
            tgt_token_ids, tgt_lens = get_seq_features_from_line(tgt_tokens=target_tokens, vocab=vocab)
            g.tgt_token_ids = tgt_token_ids
            g.tgt_token_lens = tgt_lens
            g.to(device)
            data_list.append(g)
        data_loader = DataLoader(dataset=data_list, batch_size=64, shuffle=False)
        dict_data_loader = MP_Matrix_Creator(data_loader, device)

        #Encode and predict
        batches = list(range(len(dict_data_loader)))
        y_p = []
        z_p = []
        all_reconstructions = []
        with torch.no_grad():
            for i, batch in enumerate(batches):
                data = dict_data_loader[str(batch)][0]
                data.to(device)
                dest_is_origin_matrix = dict_data_loader[str(batch)][1]
                dest_is_origin_matrix.to(device)
                inc_edges_to_atom_matrix = dict_data_loader[str(batch)][2]
                inc_edges_to_atom_matrix.to(device)

                # Perform a single forward pass.
                reconstruction, _, _, z, y = model.inference(data=data, device=device, dest_is_origin_matrix=dest_is_origin_matrix, inc_edges_to_atom_matrix=inc_edges_to_atom_matrix, sample=False, log_var=None)
                y_p.append(y.cpu().numpy())
                z_p.append(z.cpu().numpy())
                reconstruction_strings = [combine_tokens(tokenids_to_vocab(reconstruction[sample][0].tolist(), vocab), tokenization=tokenization) for sample in range(len(reconstruction))]
                all_reconstructions.extend(reconstruction_strings)
        #Return the predictions from the encoded latents
        y_p_flat = [sublist.tolist() for array_ in y_p for sublist in array_]
        z_p_flat = [sublist.tolist() for array_ in z_p for sublist in array_]
        self.modified_solution = z_p_flat

        return y_p_flat, z_p_flat, all_reconstructions, dict_data_loader


algorithm = NSGA2(pop_size=pop_size,
                  sampling=sampling,
                  crossover=crossover,
                  repair=correctSamplesRepair(model),
                  mutation=mutation,
                  eliminate_duplicates=True)

# Optimize the problem
all_solutions = []
best_solutions = []
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    verbose=True,
)

# Access the results
best_solution = res.X
#best_mod_solution = res.X_mod
best_fitness = res.F
results_custom = problem.results_custom
 

with open(dir_name+'res_optimization_GA_correct_'+str(objective_type), 'wb') as f:
    pickle.dump(res, f)
with open(dir_name+'optimization_results_custom_GA_correct_'+str(objective_type)+'.pkl', 'wb') as f:
    pickle.dump(results_custom, f)
with open(dir_name+'optimization_results_custom_GA_correct_'+str(objective_type)+'.txt', 'w') as fl:
     print(results_custom, file=fl)

#convergence = res.algorithm.termination
with open(dir_name+'res_optimization_GA_correct_'+str(objective_type), 'rb') as f:
    res = pickle.load(f)

with open(dir_name+'optimization_results_custom_GA_correct_'+str(objective_type)+'.pkl', 'rb') as f:
    results_custom = pickle.load(f)

# Calculate distances between the BO and reencoded latents
Latents_RE = []
pred_RE = []
decoded_mols= []
pred_RE_corrected = []

for idx, (pop, res) in enumerate(list(results_custom.items())):
    population= int(pop)
    # loop through population
    pop_size = len(list(res["objective"]))
    for point in range(pop_size):
        L_re=res["latents_reencoded"][point]
        Latents_RE.append(L_re)
        pred_RE.append(res["predictions"][point])
        pred_RE_corrected.append(res["predictions_doublecorrect"][point])
        decoded_mols.append(res["string_decoded"][point])

import matplotlib.pyplot as plt

iterations = range(len(pred_RE))
EA_re= [arr[0].cpu() for arr in pred_RE]
IP_re = [arr[1].cpu() for arr in pred_RE]
EA_re_c= [arr[0] for arr in pred_RE_corrected]
IP_re_c = [arr[1] for arr in pred_RE_corrected]

# Create plot
plt.figure(0)

plt.plot(iterations, EA_re, label='EA (RE)')
plt.plot(iterations, IP_re, label='IP (RE)')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.savefig(dir_name+'GA_objectives_correct.png',  dpi=300)
plt.close()

import math 
def indices_of_improvement(values):
    indices_of_increases = []

    # Initialize the highest value and its index
    highest_value = values[0]
    highest_index = 0

    # Iterate through the values
    for i, value in enumerate(values):
        # If the current value is greater than the previous highest value
        if value < highest_value:
            highest_value = value  # Update the highest value
            highest_index = i      # Update the index of the highest value
            indices_of_increases.append(i)  # Save the index of increase

    return indices_of_increases

def top_n_molecule_indices(objective_values, decoded_mols, n_idx=10):
    # Get the indices of 20 molecules with the best objective values
    # Pair each value with its index
    # Filter out NaN values and keep track of original indices
    filtered_indexed_values = [(index, value) for index, value in enumerate(objective_values) if not math.isnan(value)]
    # Sort the indexed values by the value in ascending order and take n_idx best ones
    sorted_filtered_indexed_values = sorted(filtered_indexed_values, key=lambda x: x[1], reverse=False)
    _best_mols = []
    best_mols_count = {}
    top_idxs = []
    for index, value in sorted_filtered_indexed_values: 
        if not decoded_mols[index] in _best_mols: 
            top_idxs.append(index)
            best_mols_count[decoded_mols[index]]=1
            _best_mols.append(decoded_mols[index])
        else:
            best_mols_count[decoded_mols[index]]+=1
        if len(top_idxs)==20:
            break

    return top_idxs, best_mols_count

# Extract data for the curves
if objective_type=='mimick_peak':
    objective_values = [(np.abs(arr.cpu()[0]+2)+np.abs(arr.cpu()[1]-1.2)) for arr in pred_RE]
    objective_values_c = [(np.abs(arr[0]+2)+np.abs(arr[1]-1.2)) for arr in pred_RE_corrected]
elif objective_type=='mimick_best':
    objective_values = [(np.abs(arr.cpu()[0]+2.64)+np.abs(arr.cpu()[1]-1.61)) for arr in pred_RE]
    objective_values_c = [(np.abs(arr[0]+2.64)+np.abs(arr[1]-1.61)) for arr in pred_RE_corrected]
elif objective_type=='EAmin': 
    objective_values = [arr.cpu()[0]+np.abs(arr.cpu()[1]-1) for arr in pred_RE]
    objective_values_c = [arr[0]+np.abs(arr[1]-1) for arr in pred_RE_corrected]
elif objective_type =='max_gap':
    objective_values = [arr.cpu()[0]-arr.cpu()[1] for arr in pred_RE]
    objective_values_c = [arr[0]-arr[1] for arr in pred_RE_corrected]

indices_of_increases = indices_of_improvement(objective_values)


EA_re_imp = [EA_re[i] for i in indices_of_increases]
IP_re_imp = [IP_re[i] for i in indices_of_increases]
best_z_re = [Latents_RE[i] for i in indices_of_increases]
best_mols = {i+1: decoded_mols[i] for i in indices_of_increases}
best_props = {i+1: [EA_re[i],IP_re[i]] for i in indices_of_increases}
with open(dir_name+'best_mols_GA_correct_IP5_'+str(objective_type)+'.txt', 'w') as fl:
    print(best_mols, file=fl)
    print(best_props, file=fl)

top_20_indices, top_20_mols = top_n_molecule_indices(objective_values, decoded_mols, n_idx=20)
best_mols_t20 = {i+1: decoded_mols[i] for i in top_20_indices}
best_props_t20 = {i+1: [EA_re[i], IP_re[i]] for i in top_20_indices}
best_props_t20_c = {i+1: [EA_re_c[i], IP_re_c[i]] for i in top_20_indices}
best_objs_t20 = {i+1: objective_values[i] for i in top_20_indices}
best_objs_t20_c = {i+1: objective_values_c[i] for i in top_20_indices}
with open(dir_name+'top20_mols_GA_correct_IP5_'+str(objective_type)+'.txt', 'w') as fl:
    print(best_mols_t20, file=fl)
    print(best_props_t20, file=fl)
    print(best_props_t20_c, file=fl)
    print(best_objs_t20, file=fl)
    print(best_objs_t20_c, file=fl)
    print(top_20_mols, file=fl)

