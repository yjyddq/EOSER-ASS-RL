#!/bin/bash
source /mnt/shared-storage-user/yangjingyi/anaconda3/bin/activate
conda activate EOSER_ASS_RL
cd /mnt/shared-storage-user/yangjingyi/EOSER_ASS_RL/eval
export HF_HOME=/mnt/shared-storage-user/yangjingyi/huggingface
export LOGDIR=/mnt/shared-storage-user/yangjingyi/EOSER_ASS_RL/eval/logs
export OMP_NUM_THREADS=1

# Configuration variables
GPU_IDS=(0 1 2 3 4 5 6 7)

# Arrays of tasks and generation lengths
TASKS=("countdown" "sudoku" "math" "gsm8k")
GEN_LENGTHS=(256)
block_length=1
diffusion_steps=8
strategy="eoser_ass" # ['eoser_ass', 'ass', 'semi-ar_lcr', 'semi-ar_rr', 'diffusion_lcr', 'diffusion_rr']

# Set GPU IDs from command line if provided
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Select adapter path per task
    case "$task" in
      countdown)
        adapter_path=""
        ;;
      sudoku)
        adapter_path=""
        ;;
      math)
        adapter_path=""
        ;;
      gsm8k)
        adapter_path=""
        ;;
      *)
        echo "No adapter configured for task: $task" >&2
        exit 1
        ;;
    esac

    if [ ! -d "$adapter_path" ] && [ ! -f "$adapter_path" ]; then
      echo "Adapter path does not exist: $adapter_path" >&2
      exit 1
    fi

    # Set batch size based on generation length
    if [ "$gen_length" -eq 512 ]; then
      batch_size=4
    else
      batch_size=8
    fi

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    MASTER_PORT=$((29411 + RANDOM % 1000))

    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size, strategy=$strategy, block_length=$block_length, diffusion_steps=$diffusion_steps"
    
    torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval.py \
      --dataset $task \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --block_length $block_length \
      --diffusion_steps $diffusion_steps \
      --output_dir "/mnt/shared-storage-user/yangjingyi/EOSER_ASS_RL/eval/cj_${task}_${strategy}_${gen_length}_${block_length}_${diffusion_steps}_${TIMESTAMP}_eval_results" \
      --model_path "GSAI-ML/LLaDA-8B-Instruct" \
      --strategy $strategy \
      --adapter_path "$adapter_path" \
      >> ${LOGDIR}/eval-dj-${task}-${strategy}-${gen_length}-${block_length}-${diffusion_steps}-${TIMESTAMP}.out \
      2>> ${LOGDIR}/eval-dj-${task}-${strategy}-${gen_length}-${block_length}-${diffusion_steps}-${TIMESTAMP}.err &
      

    # python3 parse_and_get_acc.py --directory "/mnt/shared-storage-user/yangjingyi/EOSER_ASS_RL/eval/cj_${task}_${strategy}_${gen_length}_${block_length}_${diffusion_steps}_${TIMESTAMP}_eval_results"
  done
done


echo "All evaluations completed!"
