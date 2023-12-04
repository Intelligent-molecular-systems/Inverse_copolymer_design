# %% Packages
import time
from datetime import datetime
import sys
import random
from G2S import *
# deep learning packages
import torch
from data_utils import *
from statistics import mean
import os
import pickle
import math


class EarlyStopping:
    def __init__(self, dir, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_dir = dir
    
    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            torch.save(model.state_dict(), self.save_dir + "/model_best_loss.pth")
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            torch.save(model.state_dict(), self.save_dir + "/model_best_loss.pth")
            self.counter = 0


def train(dict_train_loader, global_step, monotonic_step):
    # shuffle batches every epoch
    order_batches = list(range(len(dict_train_loader)))
    random.shuffle(order_batches)

    ce_losses = []
    total_losses = []
    kld_losses = []
    accs = []

    model.train()
    # Iterate in batches over the training dataset.
    for i, batch in enumerate(order_batches):
        if model_config['beta']=="schedule":
            # determine beta at time step t
            if global_step >= len(beta_schedule):
                beta_t = 1.0 +0.01*monotonic_step
                monotonic_step+=1
                print("Using monotonic schedule")
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
        loss, recon_loss, kl_loss, acc, predictions, target, _ = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)
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
        if i % 10 == 0:
            print(f"Batch [{i} / {len(order_batches)}]")
            print(recon_loss.item(), loss.item(), kl_loss.item(), acc.item())
            print("beta:"+str(beta_t))
        
        global_step += 1
        
    return model, ce_losses, total_losses, kld_losses, accs, global_step, monotonic_step


def test(dict_loader):
    batches = list(range(len(dict_loader)))
    ce_losses = []
    total_losses = []
    kld_losses = []
    accs = []

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
            loss, recon_loss, kl_loss, acc, predictions, target, _ = model(data, dest_is_origin_matrix, inc_edges_to_atom_matrix, device)

            ce_losses.append(recon_loss.item())
            total_losses.append(loss.item())
            kld_losses.append(kl_loss.item())
            accs.append(acc.item())
        
    return ce_losses, total_losses, kld_losses, accs

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

# Model config and vocab
vocab = load_vocab(vocab_file='poly_smiles_vocab.txt')
model_config = {
    "embedding_dim": 32, # needs to be hidden dimension
    "beta": "schedule",
    "decoder_num_layers": 4,
    "num_attention_heads":4,
    'batch_size': 64,
    'epochs': 40,
    'hidden_dimension': 32, #300
    'learning_rate': 1e-3,
    'es_patience': 5,
    'loss': "focal", # or ce
}
batch_size = model_config['batch_size']
epochs = model_config['epochs']
hidden_dimension = model_config['hidden_dimension']
loss = model_config['loss']

# %% Call data

dict_train_loader = torch.load('dict_train_loader.pt')
dict_val_loader = torch.load('dict_val_loader.pt')
dict_test_loader = torch.load('dict_test_loader.pt')

num_train_graphs = len(list(dict_train_loader.keys())[
    :-2])*batch_size + dict_train_loader[list(dict_train_loader.keys())[-1]][0].num_graphs
num_node_features = dict_train_loader['0'][0].num_node_features
num_edge_features = dict_train_loader['0'][0].num_edge_features

assert dict_train_loader['0'][0].num_graphs == batch_size, 'Batch_sizes of data and model do not match'

# %% Create an instance of the G2S model

model = G2S_PP(num_node_features,num_edge_features,hidden_dimension,device,model_config,vocab)
model.to(device)
print(model)

#untrained_state_dict = copy.deepcopy(model.state_dict())
print('Randomly initialized weights')
# weight initialization
# takes in a module and applies the specified weight initialization
""" def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        if m.bias is not None:
            m.bias.data.fill_(0)


model.apply(weights_init_uniform) """
n_iter = int(epochs/2 * num_train_graphs/batch_size)
# Beta scheduling function from Optimus paper 
def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=5, ratio_increase=0.5, ratio_zero=0.3):
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
    return L 

beta_schedule = frange_cycle_zero_linear(n_iter=n_iter)

# %%# %% Train

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Early stopping callback
# Log directory creation
dt_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
directory_path = os.path.join(os.getcwd(),'logfiles/', dt_string)
os.makedirs(directory_path)

es_patience = model_config['es_patience']
earlystopping = EarlyStopping(dir=directory_path, patience=es_patience)

print(f'STARTING TRAINING')
# Prepare dictionaries for training or load checkpoint
model_file = 'Checkpoints/Model_ppVAE_h='+str(model_config['hidden_dimension'])+'_decL='+str(model_config['decoder_num_layers'])+'_nheads='+str(model_config['num_attention_heads'])+'_beta='+str(model_config['beta'])+'.pt'
if os.path.isfile(os.path.join(os.getcwd(),model_file)):
    print("Loading model from checkpoint")

    checkpoint = torch.load(os.path.join(os.getcwd(),model_file))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_cp = checkpoint['epoch']
    train_loss_dict = checkpoint['loss_dict']
    val_loss_dict = checkpoint['val_loss_dict']
    if model_config['beta'] == "schedule":
        global_step = checkpoint['global_step']
        monotonic_step = checkpoint['monotonic_step']
    # increasing beta more: use monotonic steps from beta=1 to increase 0.01 per optimizer step
    #TODO: also monotonic step in checkpoint
    else:
        monotonic_step = 0
        global_step = 0
        # load monotonic checkpoint as well

else: 
    train_loss_dict = {}
    val_loss_dict = {}
    epoch_cp = 0
    global_step = 0
    monotonic_step = 0

for epoch in range(epoch_cp, epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    t1 = time.time()
    model, train_ce_losses, train_total_losses, train_kld_losses, train_accs, global_step, monotonic_step = train(dict_train_loader, global_step, monotonic_step)
    t2 = time.time()
    print(f'epoch time: {t2-t1}\n')
    val_ce_losses, val_total_losses, val_kld_losses, val_accs = test(dict_val_loader)
    train_loss = mean(train_total_losses)
    val_loss = mean(val_total_losses)
    train_kld_loss = mean(train_kld_losses)
    val_kld_loss = mean(val_kld_losses)
    train_acc = mean(train_accs)
    val_acc = mean(val_accs)
    
    # Early stopping check
    earlystopping(val_loss, model)
    if earlystopping.early_stop:
        print("Early stopping!")
        break

    print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f} | Train KLD: {train_kld_loss:.5f} \n\
                         | Val Loss: {val_loss:.5f} | Val KLD: {val_kld_loss:.5f}\n")
    train_loss_dict[epoch] = (train_loss, train_kld_loss, train_acc)
    val_loss_dict[epoch] = (val_loss, val_kld_loss, val_acc)

    if epoch % 2 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_dict': train_loss_dict,
            'val_loss_dict': val_loss_dict,
            'model_config':model_config,
            'global_step':global_step,
            'monotonic_step':monotonic_step,
            }, os.path.join(os.getcwd(),model_file))

# save model weights after training
# TODO: names for saved weights
torch.save(model.state_dict(),
            'Models_Weights/weights_training='+str(1)+'.pt')
# Save the training loss values
with open('./train_loss.pkl', 'wb') as file:
    pickle.dump(train_loss_dict, file)
 
# Save the validation loss values
with open('./val_loss.pkl', 'wb') as file:
    pickle.dump(val_loss_dict, file)

print('Done!\n')
#experiment.end()


# %%
