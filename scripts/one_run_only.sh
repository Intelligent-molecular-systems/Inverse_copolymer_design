#!/bin/sh


beta="schedule"
augmented="augmented"
add_latent=1
loss="wce"
tokenization="RT_tokenized"
augmented="augmented_enum"
seed=3 
epsilon=0.01
max_beta=1
sbatch model_run.sbatch -a ${augmented} -s ${seed} -l ${loss} -b ${beta} -maxb ${max_beta} -eps ${epsilon} -t ${tokenization} -al ${add_latent} 

