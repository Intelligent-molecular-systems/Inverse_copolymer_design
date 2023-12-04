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
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt





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

# Call data
dataset_type = "val"


dict_test_loader = torch.load('dict_'+dataset_type+'_loader.pt')

num_node_features = dict_test_loader['0'][0].num_node_features
num_edge_features = dict_test_loader['0'][0].num_edge_features

# Load model
# Create an instance of the G2S model from checkpoint


checkpoint = torch.load(os.path.join(os.getcwd(),'Checkpoints/Model_ppVAE_h=32_decL=4_nheads=4_beta=schedule.pt'), map_location=torch.device('cpu'))
model_config = checkpoint["model_config"]
batch_size = model_config['batch_size']
hidden_dimension = model_config['hidden_dimension']
vocab = load_vocab(vocab_file='poly_smiles_vocab.txt')
model = G2S_PP(num_node_features,num_edge_features,hidden_dimension,device,model_config,vocab)
model.load_state_dict(checkpoint['model_state_dict'])
#model.load_state_dict(torch.load('Models_Weights/weights_training='+str(1)+'.pt', map_location=torch.device('cpu')))
model.to(device)

# Directory to save results
dir_name= 'Model_ppVAE_h='+str(model_config['hidden_dimension'])+'_decL='+str(model_config['decoder_num_layers'])+'_nheads='+str(model_config['num_attention_heads'])+'_beta='+str(model_config['beta'])+'/'
if not os.path.exists(dir_name):
   os.makedirs(dir_name)

loss_dict = checkpoint['loss_dict']
val_loss_dict = checkpoint['val_loss_dict']
tot_loss_train = [loss[0] for loss in loss_dict.values()]
kl_loss_train = [loss[1] for loss in loss_dict.values()]
acc_train = [loss[2] for loss in loss_dict.values()]
tot_loss_val = [loss[0] for loss in val_loss_dict.values()]
kl_loss_val = [loss[1] for loss in val_loss_dict.values()]
acc_val = [loss[2] for loss in val_loss_dict.values()]
epochs = range(1,len(acc_train)+1)

# PLOTS
plt.figure(0)
fig, ax1 = plt.subplots()
lns1 = ax1.plot(epochs, tot_loss_train, '-', c="blue", label = "Total loss")
lns2 = ax1.plot(epochs, kl_loss_train, '--', c="blue", label = "KL loss", )
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
lns3 = ax2.plot(epochs, acc_train, '-', c="green", label="accuracy")
ax2.set_ylabel("Accuracy")
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
fig.savefig(dir_name+'train_loss_curves.png')



plt.figure(1)
fig, ax1 = plt.subplots()
lns1 = ax1.plot(epochs, tot_loss_val, '-', c="blue", label = "CE reconstruction loss")
lns2 = ax1.plot(epochs, kl_loss_val, '--', c="blue", label = "KL loss", )
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
lns3 = ax2.plot(epochs, acc_val, '-', c="green", label="accuracy")
ax2.set_ylabel("Accuracy")
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
fig.savefig(dir_name+'val_loss_curves.png')
