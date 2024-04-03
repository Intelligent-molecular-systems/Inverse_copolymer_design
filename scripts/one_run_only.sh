#!/bin/sh


beta="schedule"
augmented="augmented"
add_latent=1
loss="wce"
tokenization="RT_tokenized"
seed=1
epsilon=1.0
max_beta=0.0005
ppguided=1

sbatch model_run.sbatch -a ${augmented} -s ${seed} -l ${loss} -b ${beta} -maxb ${max_beta} -eps ${epsilon} -t ${tokenization} -al ${add_latent} -ppg ${ppguided}

