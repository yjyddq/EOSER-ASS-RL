set -ex
source /mnt/shared-storage-user/yangjingyi/anaconda3/bin/activate
conda activate EOSER-ASS-RL
cd /mnt/shared-storage-user/yangjingyi/EOSER-ASS-RL/cj-grpo

export BNB_FORCE_CPU=1
export OMP_NUM_THREADS=1
export WANDB_ENTITY=
export WANDB_API_KEY=
export WANDB_MODE=

LOGDIR=/mnt/shared-storage-user/yangjingyi/EOSER-ASS-RL/cj-grpo/logs
SAVEDIR=/mnt/shared-storage-user/yangjingyi/EOSER-ASS-RL/cj-grpo/checkpoints
mkdir -p $LOGDIR
mkdir -p $SAVEDIR
echo $LOGDIR
echo $SAVEDIR

# DATASETS = ("countdown" "gsm8k" "math" "sudoku")
# NUM_ITERS= (8 12 12 8)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATASET="countdown"
RUN_NAME=cj_grpo_${DATASET}_128_64_64
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
    --decoding 'eoser-ass' \
    >> ${LOGDIR}/EOSER-ASS-RL-${DATASET}-${TIMESTAMP}.out \
    2>> ${LOGDIR}/EOSER-ASS-RL-${DATASET}-${TIMESTAMP}.err &
