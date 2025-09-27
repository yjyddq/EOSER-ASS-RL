#!/bin/bash
#SBATCH -J diffu-grpo
#SBATCH -p parition
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH -N 1
#SBATCH -o /mnt/petrelfs/yangjingyi/EOSER-ASS-RL/diffu-grpo/logs/diffu-grpo-countdown-%j.out
#SBATCH -e /mnt/petrelfs/yangjingyi/EOSER-ASS-RL/diffu-grpo/logs/diffu-grpo-countdown-%j.err

export BNB_FORCE_CPU=1
export OMP_NUM_THREADS=1
export LOGDIR=logs
export SAVEDIR=checkpoints
mkdir -p $LOGDIR
mkdir -p $SAVEDIR
echo $LOGDIR
echo $SAVEDIR


DATASET="countdown"
RUN_NAME=diffu_grpo_${DATASET}
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
# ADAPTER_PATH=/mnt/petrelfs/yangjingyi/EOSER-ASS-RL/sft/outputs/llada-s1/checkpoint-2460
NUM_ITER=8

accelerate launch \
    --config_file slurm_scripts/accelerate_a100.yaml \
    --main_process_port 12346 diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME
    # --adapter_path $ADAPTER_PATH
