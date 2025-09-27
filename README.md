<div  align="center">
    <h1>Taming Masked Diffusion Language Models via Consistency Trajectory Reinforcement Learning with Fewer Decoding Step</h1>

  <span style="color:red">📢 <strong><i>If you also engaged in the research of MDLMs or RL, we welcome your suggestions. And feel free to create an issue, when you have any questions about the code.
  If you are interested in our work, please star ⭐ our repository, Thx 💕.</i></strong></span>

  <h4>
    <img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
    <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=yjyddq.EOSER-ASS-RL" />
    <img src="https://img.shields.io/github/stars/yjyddq/EOSER-ASS-RL?color=yellow" alt="Stars">
    <img src="https://img.shields.io/github/issues/yjyddq/EOSER-ASS-RL?color=red" alt="Issues">
  </h4>
</div>

<p>We propose EOS Early Rejection (EOSER) decoding and Ascending Step-Size (ASS) scheduler, which unlock the potential of MDLMs to perform full diffusion-style decoding, achieving competitive performance with fewer decoding steps. Additionally, we introduce Consistency Trajectory Group Relative Policy Optimization (CJ-GRPO) for taming MDLMs, which emphasizes the consistency between rollout trajectory and optimization trajectory. The experimental results demonstrate that the proposed EOSER and ASS mechanisms, together with CJ-GRPO, hold significant promise for effectively and efficiently taming MDLMs.</p>

![Motivation](media/Motivation.jpg)

<div align="center">
  <hr width="100%">
</div>

## 📢 Updates

* TBD: Our paper will coming soon.
* 09-28-2025: We released the code of Taming Masked Diffusion Language Models via Consistency Trajectory Reinforcement Learning with Fewer Decoding Step.

<div align="center">
  <hr width="100%">
</div>


## ⚙️ Environment Setup

To setup the environment, run;
```
conda env create -f env.yml
conda activate EOSER-ASS-RL
```

## 🚀 SFT

The code for Supervised Fine-Tuning (SFT) is sourced from the [d1](https://github.com/dllm-reasoning/d1/tree/main/SFT).

sft results can be reproduced with the command,
```bash
# First go to the sft directory
cd sft

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ddp_config.yaml --main_process_port 29500 --num_processes 2 sft_train.py --grad_accum_steps 4 --batch_size 1 --num_epochs 20 
# this results in effective batch size of 8 = 1 * 2 * 4, where 2 is the number of gpus.
```

## 🚀 Decoding Strategy: EOSER and ASS Scheduler 

The code is inside the `eval/generate.py`.

![Decoding](media/Decoding.jpg)

## 🚀 Consistency or InConsistency Trajectory Group Relative Policy Optimization

The code is inside the `cj-grpo` directory.

- `cj-grpo/slurm_scripts` contains the slurm scripts we used to run the CJ-GRPO experiments
- Example bash script for running the CJ-GRPO experiment:
  ```bash
  cd cj-grpo
  
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh
  ```

The code of one-step optimization (from x<sub>0</sub> to x<sub>S</sub>) is inside the `one-step-grpo` directory.

- `one-step-grpo/slurm_scripts` contains the slurm scripts we used to run the one-step optimization experiments
- Example bash script for running the one-step-GRPO experiment:
  ```bash
  cd one-step-grpo
  
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh
  ```

The code of one-step optimization with prompt masking, i.e., d1 (from x'<sub>S</sub> to x<sub>S</sub>) is inside the `diffu-grpo` directory.

- `diffu-grpo/slurm_scripts` contains the slurm scripts we used to run the one-step optimization with prompt masking experiments
- Example bash script for running the one-step-GRPO with prompt masking experiment:
  ```bash
  cd diffu-grpo
  
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh
  ```

The difference betwee MDLMs and AR LLMs during RL optimization (e.g., GRPO).

![CJ-GRPO](media/CJ-GRPO.jpg)

Algorithm Optimization Pipeline:

![Algorithm](media/Algorithm.jpg)


## 🚀 Evaluation

The evaluation code is inside the `eval` directory.

- Run with `bash run_eval.sh`
- The evaluation file will only save the generations, use `python parse_and_get_acc.py` to print the accuracy.


## 🔗 Citation

If you find this work useful, please consider citing:

```bibtex
TBD
```

## 🙏 Acknowledgements

Parts of the codes are borrowed from [d1](https://github.com/dllm-reasoning/d1). Sincere thanks to their wonderful works.