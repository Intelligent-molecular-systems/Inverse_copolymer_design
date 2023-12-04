#!/bin/sh


beta="schedule"
augmented="augmented"
add_latent=1
for loss in "ce" "wce"
do
	for tokenization in "oldtok" "RT_tokenized"
	do
		for seed in 42 43 44 45 46 47 48 49 50 51
		do
			sbatch model_run.sbatch -a ${augmented} -s ${seed} -l ${loss} -b ${beta} -t ${tokenization} -al ${add_latent}
		done
	done
done

beta="schedule"
augmented="augmented"
add_latent=0
loss="ce"
tokenization="oldtok"
for seed in 42 43 44 45 46 47 48 49 50 51
do
	sbatch model_run.sbatch -a ${augmented} -s ${seed} -l ${loss} -b ${beta} -t ${tokenization} -al ${add_latent}
done

beta="normalVAE"
augmented="augmented"
add_latent=1
loss="ce"
tokenization="oldtok"
for seed in 42 43 44 45 46 47 48 49 50 51
do
	sbatch model_run.sbatch -a ${augmented} -s ${seed} -l ${loss} -b ${beta} -t ${tokenization} -al ${add_latent}
done


