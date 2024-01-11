#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --gres=gpu:pascal:1
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --mail-type=END


source /opt/insy/miniconda/3.9/etc/profile.d/conda.sh
conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/gabrielvogel/tryonmt_env/
module load cuda/11.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tudelft.net/staff-bulk/ewi/insy/DBL/gabrielvogel/tryonmt_env/lib 
export CUBLAS_WORKSPACE_CONFIG=:4096:8

augmented="augmented"
seed=42
loss="wce"
beta="schedule"
tokenization="RT_tokenized"
add_latent=1
ppguided=0
dec_layers=4


while true;
do
	case "$1" in
		-a) augmented=$2; shift 2;;
		-s) seed=$2; shift 2;;
        -l) loss=$2; shift 2;;
        -b) beta=$2; shift 2;;
        -t) tokenization=$2; shift 2;;
		-al) add_latent=$2; shift 2;;
		-ppg) ppguided=$2; shift 2;;
		-decl) dec_layers=$2; shift 2;;
		--) shift; break ;;
		*) break ;;
	esac		
done


prefix="Checkpoints_new/logs/${augmented}_${beta}_${loss}_${tokenization}_addlatent${add_latent}_${seed}_ppg${ppguided}_decL${dec_layers}_"

srun python train_with_args.py --augment=${augmented} --beta=${beta} --loss=${loss} --tokenization=${tokenization} --seed=${seed} --add_latent=${add_latent} --ppguided=${ppguided} --dec_layers=${dec_layers} > ${prefix}train_out.txt 
srun python generate.py --augment=${augmented} --beta=${beta} --loss=${loss} --tokenization=${tokenization} --seed=${seed} --add_latent=${add_latent} --ppguided=${ppguided} --dec_layers=${dec_layers}  > ${prefix}gen_out.txt 
srun python test_cluster.py --augment=${augmented} --beta=${beta} --loss=${loss} --tokenization=${tokenization} --seed=${seed} --add_latent=${add_latent} --ppguided=${ppguided} --dec_layers=${dec_layers}  > ${prefix}test_out.txt 
srun python inference_cluster.py --augment=${augmented} --beta=${beta} --loss=${loss} --tokenization=${tokenization} --seed=${seed} --add_latent=${add_latent} --ppguided=${ppguided} --dec_layers=${dec_layers}  > ${prefix}inf_out.txt 






