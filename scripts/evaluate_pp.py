import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)
import pandas as pd
from statistics import mean
import numpy as np
import argparse
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity



parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original, augmented_canonical", default="original", choices=["augmented","augmented_old", "original", "augmented_canonical", "augmented_enum"])
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
data_augment = "old"

# Directory to save results
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'/'
    # Directory to save results
dir_name=  os.path.join(main_dir_path,'Checkpoints/', model_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


with open(dir_name+'y1_all_'+dataset_type+'.npy', 'rb') as f:
    y1_all = np.load(f)
with open(dir_name+'y2_all_'+dataset_type+'.npy', 'rb') as f:
    y2_all = np.load(f)
with open(dir_name+'yp_all_'+dataset_type+'.npy', 'rb') as f:
    yp_all = np.load(f)

y1_all=list(y1_all)
y2_all=list(y2_all)
yp1_all = [yp[0] for yp in yp_all]
yp2_all = [yp[1] for yp in yp_all]

print(len(yp1_all), len(y1_all))

def calculate_rmse_r2(real, predicted):
    # Remove Nones from real and corresponding values from predicted
    real_values = [r for r, p in zip(real, predicted) if not np.isnan(r)]
    predicted_values = [p for r, p in zip(real, predicted) if not np.isnan(r)]

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(real_values, predicted_values))

    # Calculate R-squared
    r2 = r2_score(real_values, predicted_values)

    return rmse, r2


rmse_EA, r2_EA = calculate_rmse_r2(y1_all, yp1_all)
print(rmse_EA,r2_EA)
rmse_IP, r2_IP = calculate_rmse_r2(y2_all, yp2_all)
print(rmse_IP,r2_IP)


from matplotlib.offsetbox import AnchoredText


real = [r for r, p in zip(y1_all, yp1_all) if not np.isnan(r)]
predicted = [p for r, p in zip(y1_all, yp1_all) if not np.isnan(r)]

plt.rcParams.update({'font.size': 18})  # Apply to all text elements

plt.figure(1)
plt.scatter(real, predicted, color='blue', s=0.1)
plt.plot(real, real, color='black', linestyle='--')
plt.title('Parity Plot')
plt.xlabel('Real Values in eV')
plt.ylabel('Predicted Values in eV')
plt.title('Electron affinity')
plt.grid(True)
textstr = f'RMSE: {rmse_EA:.3f}\nR²: {r2_EA:.3f}'
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=16,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
plt.show()
plt.savefig(dir_name+'parity_EA.png', dpi=300, bbox_inches='tight')

real = [r for r, p in zip(y2_all, yp2_all) if not np.isnan(r)]
predicted = [p for r, p in zip(y2_all, yp2_all) if not np.isnan(r)]
plt.figure(2)
plt.scatter(real, predicted, color='blue', s=0.1)
plt.plot(real, real, color='black', linestyle='--')
plt.title('Parity Plot')
plt.xlabel('Real Values in eV')
plt.ylabel('Predicted Values in eV')
plt.title('Ionization potential')
plt.grid(True)
textstr = f'RMSE: {rmse_IP:.3f}\nR²: {r2_IP:.3f}'
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=16,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
plt.show()
plt.savefig(dir_name+'parity_IP.png', dpi=300, bbox_inches='tight')


# Do a KDE to check if the distributions of properties are similar (predicted vs. real lables)
""" y1 """
plt.figure(3)
real_distribution = np.array([r for r, p in zip(y1_all, yp1_all) if not np.isnan(r)])
augmented_distribution = np.array([p for r, p in zip(y1_all, yp1_all) if np.isnan(r)])

# Reshape the data
real_distribution = real_distribution.reshape(-1, 1)
augmented_distribution = augmented_distribution.reshape(-1, 1)

# Define bandwidth (bandwidth controls the smoothness of the kernel density estimate)
bandwidth = 0.1

# Fit kernel density estimator for real data
kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_real.fit(real_distribution)
# Fit kernel density estimator for augmented data
kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_augmented.fit(augmented_distribution)

# Create a range of values for the x-axis
x_values = np.linspace(min(np.min(real_distribution), np.min(augmented_distribution)), max(np.max(real_distribution), np.max(augmented_distribution)), 1000)
# Evaluate the KDE on the range of values
real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))

# Plotting
plt.plot(x_values, real_density, label='Real Data')
plt.plot(x_values, augmented_density, label='Augmented Data')
plt.xlabel('EA (eV)')
plt.ylabel('Density')
plt.title('Kernel Density Estimation (Electron affinity)')
plt.legend()
plt.show()
plt.savefig(dir_name+'KDEsy1.png')

""" y2 """
plt.figure(4)
real_distribution = np.array([r for r, p in zip(y2_all, yp2_all) if not np.isnan(r)])
augmented_distribution = np.array([p for r, p in zip(y2_all, yp2_all) if np.isnan(r)])

# Reshape the data
real_distribution = real_distribution.reshape(-1, 1)
augmented_distribution = augmented_distribution.reshape(-1, 1)

# Define bandwidth (bandwidth controls the smoothness of the kernel density estimate)
bandwidth = 0.1

# Fit kernel density estimator for real data
kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_real.fit(real_distribution)
# Fit kernel density estimator for augmented data
kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde_augmented.fit(augmented_distribution)

# Create a range of values for the x-axis
x_values = np.linspace(min(np.min(real_distribution), np.min(augmented_distribution)), max(np.max(real_distribution), np.max(augmented_distribution)), 1000)
# Evaluate the KDE on the range of values
real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))

# Plotting
plt.plot(x_values, real_density, label='Real Data')
plt.plot(x_values, augmented_density, label='Augmented Data')
plt.xlabel('IP (eV)')
plt.ylabel('Density')
plt.title('Kernel Density Estimation (Ionization potential)')
plt.legend()
plt.show()
plt.savefig(dir_name+'KDEsy2.png')