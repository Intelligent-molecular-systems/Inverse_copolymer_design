# Generate a random seed
seed=$(date +%s%N)


# Define the number of samples
num_samples_max_beta=8

# Define the minimum and maximum values
min_max_beta=0.0001
max_max_beta=1.0

# Generate a sequence of numbers in the log space
seq_max_beta_log=$(awk -v min=$min_max_beta -v max=$max_max_beta -v num_samples=$num_samples_max_beta 'BEGIN{for(i=1;i<=num_samples;i++) print exp(log(min) + (log(max)-log(min))*i/num_samples)}')
shuffled_seq_max_beta=$(shuf -n $num_samples_max_beta -e $seq_max_beta_log)

# Define the number of samples
num_samples_max_alpha=4
# Define the minimum and maximum values
min_max_alpha=0.01
max_max_alpha=1.0

# Generate a sequence of numbers in the log space
seq_max_alpha_log=$(awk -v min=$min_max_alpha -v max=$max_max_alpha -v num_samples=$num_samples_max_beta 'BEGIN{for(i=1;i<=num_samples;i++) print exp(log(min) + (log(max)-log(min))*i/num_samples)}')
shuffled_seq_max_alpha=$(shuf -n $num_samples_max_beta -e $seq_max_beta_log)



beta="schedule"
augmented="augmented_old"
add_latent=1
loss="wce"
tokenization="RT_tokenized"
epsilon=1.0
for seed in 3 4 
do
	for epsilon in 1 0.1 0.01
	do
		for max_beta in $shuffled_seq_max_beta
		do
			#sbatch model_run.sbatch -a ${augmented} -s ${seed} -l ${loss} -b ${beta} -maxb ${max_beta} -maxa ${max_alpha} -eps ${epsilon} -t ${tokenization} -al ${add_latent} 
		done
	done
done