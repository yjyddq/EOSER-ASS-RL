#!/bin/bash
#SBATCH -J cj-grpo
#SBATCH -p parition
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH -N 1
#SBATCH -o /mnt/petrelfs/yangjingyi/EOSER-ASS-RL/cj-grpo/logs/cj-grpo-sudoku-%j.out
#SBATCH -e /mnt/petrelfs/yangjingyi/EOSER-ASS-RL/cj-grpo/logs/cj-grpo-sudoku-%j.err

export BNB_FORCE_CPU=1
export OMP_NUM_THREADS=1
export LOGDIR=logs
export SAVEDIR=checkpoints
mkdir -p $LOGDIR
mkdir -p $SAVEDIR
echo $LOGDIR
echo $SAVEDIR

# DATASETS = ("countdown" "gsm8k" "math" "sudoku")
# NUM_ITERS= (8 12 12 8)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATASET="sudoku"
GEN_LENGTHS=128
BLOCK_LENGTH=64
DIFFUSION_STEPS=32
RUN_NAME=cj_grpo_${DATASET}_${GEN_LENGTHS}_${BLOCK_LENGTH}_${DIFFUSION_STEPS}_${TIMESTAMP}
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
NUM_ITER=8 # number of policy gradient inner updates iterations

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12346 cj_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir ${SAVEDIR}/$RUN_NAME \
    --max_completion_length ${GEN_LENGTHS} \
    --block_length ${BLOCK_LENGTH} \
    --block_num ${BLOCK_LENGTH} \
    --diffusion_steps ${DIFFUSION_STEPS} \
    --decoding 'eoser-ass' \
    >> ${LOGDIR}/cj-${DATASET}-${GEN_LENGTHS}-${BLOCK_LENGTH}-${DIFFUSION_STEPS}-${TIMESTAMP}.out \
    2>> ${LOGDIR}/cj-${DATASET}-${GEN_LENGTHS}-${BLOCK_LENGTH}-${DIFFUSION_STEPS}-${TIMESTAMP}.err &
            
