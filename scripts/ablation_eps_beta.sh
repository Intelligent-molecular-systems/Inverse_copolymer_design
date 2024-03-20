#!/bin/sh


beta="schedule"
augmented="augmented"
add_latent=1
loss="wce"
tokenization="RT_tokenized"
augmented="augmented"
for seed in 3 4 
do
	for epsilon in 1 0.1 0.01
	do
		for max_beta in 1 0.1 0.01 0.0025
		do
			sbatch model_run.sbatch -a ${augmented} -s ${seed} -l ${loss} -b ${beta} -maxb ${max_beta} -eps ${epsilon} -t ${tokenization} -al ${add_latent} 
		done
	done
done
