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
import numpy as np



# Directory to save results
accs_train=[]
accs_val=[]
for s in [42,43,44,45,46]:
    dir_name = os.path.join(os.getcwd(),'Checkpoints_firstsubmission/Model_beta=schedule_loss=wce_augment=augmented_tokenization=oldtok_AE_warmup=False_init=random_seed='+str(s)+'/')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    try: 
        with open(dir_name+'train_loss.pkl', 'rb') as f:
            train_loss_dict=pickle.load(f)
        with open(dir_name+'val_loss.pkl', 'rb') as f:
            val_loss_dict=pickle.load(f)
        tot_loss_train = [mean(loss[0]) for loss in train_loss_dict.values()] #mean for each epoch
        kl_loss_train = [mean(loss[1]) for loss in train_loss_dict.values()]
        acc_train = [mean(loss[2]) for loss in train_loss_dict.values()]
        tot_loss_val = [mean(loss[0]) for loss in val_loss_dict.values()]
        kl_loss_val = [mean(loss[1]) for loss in val_loss_dict.values()]
        acc_val = [mean(loss[2]) for loss in val_loss_dict.values()]

        accs_train.append(acc_train[-1])
        accs_val.append(acc_val[-1])
    except: print("model with seed %s did not converge"%s)

    

print(mean(accs_val), np.std(accs_val))

# PLOTS
""" plt.figure(0)
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
fig.savefig(dir_name+'val_loss_curves.png') """