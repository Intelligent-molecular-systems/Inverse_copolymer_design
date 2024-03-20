import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)

import time
from datetime import datetime
import random
#from G2S import *
from model.G2S_clean import *
from data_processing.data_utils import *
# deep learning packages
import torch
import torch.nn as nn
from statistics import mean
import pickle
import math
import argparse


class EarlyStopping:
    def __init__(self, dir, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_dir = dir

    def __call__(self, val_loss, model_dict):
        val_loss = round(val_loss,4)
        if self.best_score is None:
            self.best_score = val_loss
            torch.save(model_dict, os.path.join(self.save_dir,"model_best_loss.pt"))
            #torch.save(model.state_dict(), self.save_dir + "/model_best_loss.pth")
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            torch.save(model_dict, os.path.join(self.save_dir,"model_best_loss.pt"))
            #torch.save(model.state_dict(), self.save_dir + "/model_best_loss.pth")
            self.counter = 0


def train(dict_train_loader, global_step, monotonic_step):
    # shuffle batches every epoch

    order_batches = list(range(len(dict_train_loader)))
    random.shuffle(order_batches)

    ce_losses = []
    total_losses = []
    kld_losses = []
    accs = []
    mses = []

    model.train()
    # Iterate in batches over the training dataset.
    for i, batch in enumerate(order_batches):
        if model_config['beta']=="schedule":
            # determine beta at time step t
            if global_step >= len(beta_schedule):
                #if model.beta <=1:
                #    beta_t = 1.0 +0.001*monotonic_step
                #    monotonic_step+=1
                #else: beta_t = model.beta #stays the same
                beta_t = model.beta #stays the same
            else:
                beta_t = beta_schedule[global_step]
        
            model.beta = beta_t

        # get graphs & matrices for MP from dictionary
        data = dict_train_loader[str(batch)][0]
        data.to(device)
        dest_is_origin_matrix = dict_train_loader[str(batch)][1]
        dest_is_origin_matrix.to(device)
        inc_edges_to_atom_matrix = dict_train_loader[str(batch)][2]
        inc_edges_to_atom_matrix.to(device)

        # Perform a single forward pass.
        loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
        #torch.save(predictions, "Predictions/predictions_batch"+str(i)+".pt")
        #torch.save(target, "Predictions/targets_batch"+str(i)+".pt")

        optimizer.zero_grad()
        loss.backward()
        # TODO: do we need the clip_grad_norm?
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        ce_losses.append(recon_loss.item())
        total_losses.append(loss.item())
        kld_losses.append(kl_loss.item())
        accs.append(acc.item())
        mses.append(mse.item())
        if i % 10 == 0:
            print(f"Batch [{i} / {len(order_batches)}]")
            print(recon_loss.item(), loss.item(), kl_loss.item(), acc.item(), mse.item())
            print("beta:"+str(model.beta))
        
        global_step += 1
        
    return model, ce_losses, total_losses, kld_losses, accs, mses, global_step, monotonic_step


def test(dict_loader):
    batches = list(range(len(dict_loader)))
    ce_losses = []
    total_losses = []
    kld_losses = []
    accs = []
    mses = []

    model.eval()
    test_loss = 0
    # Iterate in batches over the training/test dataset.
    with torch.no_grad():
        for batch in batches:
            data = dict_loader[str(batch)][0]
            data.to(device)
            dest_is_origin_matrix = dict_loader[str(batch)][1]
            dest_is_origin_matrix.to(device)
            inc_edges_to_atom_matrix = dict_loader[str(batch)][2]
            inc_edges_to_atom_matrix.to(device)

            # Perform a single forward pass.
            loss, recon_loss, kl_loss, mse, acc, predictions, target, z, y = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)

            ce_losses.append(recon_loss.item())
            total_losses.append(loss.item())
            kld_losses.append(kl_loss.item())
            accs.append(acc.item())
            mses.append(mse.item())
        
    return ce_losses, total_losses, kld_losses, accs, mses

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
parser.add_argument("--augment", help="options: augmented, original, augmented_canonical", default="original", choices=["augmented", "original", "augmented_canonical", "augmented_enum"])
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="oldtok", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32)
parser.add_argument("--beta", default=1, help="option: <any number>, schedule", choices=["normalVAE","schedule"])
parser.add_argument("--loss", default="ce", choices=["ce","wce"])
parser.add_argument("--AE_Warmup", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initialization", default="random", choices=["random", "xavier", "kaiming"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=0)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.01)
parser.add_argument("--epsilon", type=float, default=1)



args = parser.parse_args()

# First set the seed for reproducible results
seed = args.seed
torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#torch.cuda.manual_seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)
#random.seed(seed)
#np.random.seed(seed)

augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

# Model config and vocab
vocab = load_vocab(vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt')
model_config = {
    "embedding_dim": args.embedding_dim, # latent dimension needs to be embedding dimension of word vectors
    "beta": args.beta,
    "max_beta":args.max_beta,
    "epsilon":args.epsilon,
    "decoder_num_layers": args.dec_layers,
    "num_attention_heads":4,
    'batch_size': 64,
    'epochs': 100,
    'hidden_dimension': 300, #hidden dimension of nodes
    'n_nodes_pool': 10, #how many representative nodes are used for attention based pooling
    'pooling': 'mean', #mean or custom
    'learning_rate': 1e-3,
    'es_patience': 5,
    'loss': args.loss, # focal or ce
}
batch_size = model_config['batch_size']
epochs = model_config['epochs']
hidden_dimension = model_config['hidden_dimension']
embedding_dim = model_config['embedding_dim']
loss = model_config['loss']

# %% Call data
dict_train_loader = torch.load(main_dir_path+'/data/dict_train_loader_'+augment+'_'+tokenization+'.pt')
dict_val_loader = torch.load(main_dir_path+'/data/dict_val_loader_'+augment+'_'+tokenization+'.pt')
dict_test_loader = torch.load(main_dir_path+'/data/dict_test_loader_'+augment+'_'+tokenization+'.pt')

num_train_graphs = len(list(dict_train_loader.keys())[
    :-2])*batch_size + dict_train_loader[list(dict_train_loader.keys())[-1]][0].num_graphs
num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

assert dict_train_loader['0'][0].num_graphs == batch_size, 'Batch_sizes of data and model do not match'

# %% Create an instance of the G2S model
# only for wce loss we calculate the token weights from vocabulary
if model_config['loss']=="wce":
    vocab_file=main_dir_path+'/data/poly_smiles_vocab_'+augment+'_'+tokenization+'.txt'
    class_weights = token_weights(vocab_file)
    class_weights = torch.FloatTensor(class_weights)
if model_config['loss']=="ce":
    class_weights=None

# Initialize model
if args.ppguided:
    model_type = G2S_VAE_PPguided
else:
    model_type = G2S_VAE
model = model_type(num_node_features,num_edge_features,hidden_dimension,embedding_dim,device,model_config, vocab, seed, loss_weights=class_weights, add_latent=add_latent)
model.to(device)

print(model)
print('Randomly initialized weights')

n_iter = int(20 * num_train_graphs/batch_size)# 20 epochs
# Beta scheduling function from Optimus paper 
def frange_cycle_zero_linear(n_iter, start=0.0, stop=model_config['max_beta'],  n_cycle=5, ratio_increase=0.5, ratio_zero=0.3): #, beginning_zero=0.1):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else: 
                L[int(i+c*period)] = v
                v += step
            i += 1
    ## beginning zero
    if args.AE_Warmup:
        B = np.zeros(int(5*num_train_graphs/batch_size)) # for 5 epochs
        L = np.append(B,L)
    return L 

if model_config['beta'] == "schedule":
    beta_schedule = frange_cycle_zero_linear(n_iter=n_iter)
elif model_config['beta'] == "normalVAE":
    beta_schedule = np.ones(1)

# %%# %% Train

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Early stopping callback
# Log directory creation
data_augment="old"
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
directory_path = os.path.join(main_dir_path,'Checkpoints/', model_name)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

es_patience = model_config['es_patience']
earlystopping = EarlyStopping(dir=directory_path, patience=es_patience)

print(f'STARTING TRAINING')
# Prepare dictionaries for training or load checkpoint
if os.path.isfile(os.path.join(directory_path,"model_best_loss.pt")):
    print("Loading model from checkpoint")

    checkpoint = torch.load(os.path.join(directory_path,"model_best_loss.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_cp = checkpoint['epoch']
    train_loss_dict = checkpoint['loss_dict']
    val_loss_dict = checkpoint['val_loss_dict']
    if model_config['beta'] == "schedule":
        global_step = checkpoint['global_step']
        monotonic_step = checkpoint['monotonic_step']
        model.beta =  model_config['max_beta']
        #monotonic_step = 0
else: 
    train_loss_dict = {}
    val_loss_dict = {}
    epoch_cp = 0
    global_step = 0
    monotonic_step = 0



for epoch in range(epoch_cp, epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    t1 = time.time()
    model, train_ce_losses, train_total_losses, train_kld_losses, train_accs, train_mses, global_step, monotonic_step = train(dict_train_loader, global_step, monotonic_step)
    t2 = time.time()
    print(f'epoch time: {t2-t1}\n')
    val_ce_losses, val_total_losses, val_kld_losses, val_accs, val_mses = test(dict_val_loader)
    train_loss = mean(train_total_losses)
    val_loss = mean(val_total_losses)
    train_kld_loss = mean(train_kld_losses)
    val_kld_loss = mean(val_kld_losses)
    train_acc = mean(train_accs)
    val_acc = mean(val_accs)
    
    # Early stopping check, but only if the cyclical annealing schedule is already done
    if global_step >= len(beta_schedule):
        model_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_dict': train_loss_dict,
        'val_loss_dict': val_loss_dict,
        'model_config':model_config,
        'global_step':global_step,
        'monotonic_step':monotonic_step,
        }
        earlystopping(val_loss, model_dict)
        if earlystopping.early_stop:
            print("Early stopping!")
            break     
    if math.isnan(train_loss):
        print("Network diverged!")
        break

    print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f} | Train KLD: {train_kld_loss:.5f} \n\
                         | Val Loss: {val_loss:.5f} | Val KLD: {val_kld_loss:.5f}\n")
    train_loss_dict[epoch] = (train_total_losses, train_kld_losses, train_accs)
    val_loss_dict[epoch] = (val_total_losses, val_kld_losses, val_accs)


# Save the training loss values
with open(os.path.join(directory_path,'train_loss.pkl'), 'wb') as file:
    pickle.dump(train_loss_dict, file)
 
# Save the validation loss values
with open(os.path.join(directory_path,'val_loss.pkl'), 'wb') as file:
    pickle.dump(val_loss_dict, file)

print('Done!\n')
#experiment.end()


# %%
