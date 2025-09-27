
#!/bin/bash
#SBATCH -J sft_8gpu
#SBATCH -p partition
#SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH -N 1
#SBATCH -o /mnt/petrelfs/yangjingyi/EOSER-ASS-RL/sft/logs/sft_s1k-%j.out
#SBATCH -e /mnt/petrelfs/yangjingyi/EOSER-ASS-RL/sft/logs/sft_s1k-%j.err

export BNB_FORCE_CPU=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
accelerate launch \
--config_file ddp_config.yaml \
--main_process_port 29500 \
--num_processes 8 \
sft_train.py \
--train_data simplescaling/s1K \
--grad_accum_steps 4 \
--batch_size 1 \
--num_epochs 20 \
--output_dir /mnt/petrelfs/yangjingyi/EOSER-ASS-RL/sft/outputs
