<div  align="center">
    <h1><a href="https://arxiv.org/pdf/2509.23924" target="_blank">Taming Masked Diffusion Language Models via Consistency Trajectory Reinforcement Learning with Fewer Decoding Step</h1>

  <span style="color:red">📢 <strong><i>If you also engaged in the research of MDLMs or RL, we welcome your suggestions. And feel free to create an issue, when you have any questions about the code.
  If you are interested in our work, please star ⭐ our repository, Thx 💕.</i></strong></span>

  <h4>
    <img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
    <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=yjyddq.EOSER-ASS-RL" />
    <img src="https://img.shields.io/github/stars/yjyddq/EOSER-ASS-RL?style=flat-square&logo=github" alt="Stars">
    <img src="https://img.shields.io/github/issues/yjyddq/EOSER-ASS-RL?color=red" alt="Issues">
  </h4>
</div>

<p>We propose EOS Early Rejection (EOSER) decoding and Ascending Step-Size (ASS) scheduler, which unlock the potential of MDLMs to perform full diffusion-style decoding, achieving competitive performance with fewer decoding steps. Additionally, we introduce Consistency Trajectory Group Relative Policy Optimization (CJ-GRPO) for taming MDLMs, which emphasizes the consistency between rollout trajectory and optimization trajectory. The experimental results demonstrate that the proposed EOSER and ASS mechanisms, together with CJ-GRPO, hold significant promise for effectively and efficiently taming MDLMs.</p>

![Motivation](media/Motivation.jpg)

<div align="center">
  <hr width="100%">
</div>

## 📢 Updates

* 09-30-2025: We released our [paper](https://arxiv.org/pdf/2509.23924).
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

## 🚀 EOSER Decoding and ASS Scheduler 

The code is inside the `eval/generate.py`.

![Decoding](media/Decoding.jpg)

## 🚀 Consistency or InConsistency Trajectory GRPO

The code is inside the `cj-grpo` directory:

- `cj-grpo/slurm_scripts` contains the slurm scripts we used to run the CJ-GRPO experiments
- Example bash script for running the CJ-GRPO experiment:
  ```bash
  cd cj-grpo
  
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh
  ```

The code of one-step optimization starting from the fully maksed response (from x<sub>0</sub> to x<sub>S</sub>) is inside the `one-step-grpo` directory:

- `one-step-grpo/slurm_scripts` contains the slurm scripts we used to run the one-step optimization experiments
- Example bash script for running the one-step-GRPO experiment:
  ```bash
  cd one-step-grpo
  
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh
  ```

The code of one-step optimization starting from the partially masked prompt, i.e., d1 (from x'<sub>S</sub> to x<sub>S</sub>) is inside the `diffu-grpo` directory:

- `diffu-grpo/slurm_scripts` contains the slurm scripts we used to run the one-step optimization with prompt masking experiments
- Example bash script for running the one-step-GRPO with prompt masking experiment:
  ```bash
  cd diffu-grpo
  
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh
  ```

The difference betwee MDLMs and AR LLMs during RL optimization (e.g., GRPO):

![CJ-GRPO](media/CJ-GRPO.jpg)

Algorithm Optimization Pipeline:

![Algorithm](media/Algorithm.jpg)


## 🚀 Evaluation

The evaluation code is inside the `eval` directory:

- Run with `bash run_eval.sh`
- The evaluation file will only save the generations, use `python parse_and_get_acc.py` to print the accuracy.


## 🔗 Citation

If this paper or code are useful for you, please consider citing our paper:

```bibtex
@misc{yang2025tamingmaskeddiffusionlanguage,
      title={Taming Masked Diffusion Language Models via Consistency Trajectory Reinforcement Learning with Fewer Decoding Step}, 
      author={Jingyi Yang and Guanxu Chen and Xuhao Hu and Jing Shao},
      year={2025},
      eprint={2509.23924},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.23924}, 
}
```

## 🙏 Acknowledgements

Parts of the codes are borrowed from [d1](https://github.com/dllm-reasoning/d1). Sincere thanks to their wonderful works.