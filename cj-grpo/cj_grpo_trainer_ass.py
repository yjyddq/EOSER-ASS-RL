import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import wandb

if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class FlowGRPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # 添加激活检查点设置，为了防止OOM
        self.model.model.model.set_activation_checkpointing('whole_layer')  # 或其他策略

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        reversed_traj = inputs['reversed_traj']
        reversed_traj_unmask_positions = inputs['reversed_traj_unmask_positions']
        reversed_traj_old_per_token_logps = inputs['reversed_traj_old_per_token_logps']
        reversed_traj_ref_per_token_logps = inputs['reversed_traj_ref_per_token_logps']
        
        steps = len(reversed_traj_unmask_positions)

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        reversed_traj_per_token_logps = self._get_per_token_logps(model, input_ids, reversed_traj, reversed_traj_unmask_positions, logits_to_keep)

        loss = torch.tensor(0.0, device=reversed_traj_per_token_logps[0].device)
        mean_kl = torch.tensor(0.0, device=reversed_traj_per_token_logps[0].device)
        clip_ratio = torch.tensor(0.0, device=reversed_traj_per_token_logps[0].device)
        for i in range(steps):
            per_token_logps = reversed_traj_per_token_logps[i]
            # Compute the KL divergence between the model and the reference model
            if self.beta != 0.0:
                ref_per_token_logps = reversed_traj_ref_per_token_logps[i]
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                )

            # Compute the loss
            advantages = inputs["advantages"]
            old_per_token_logps = (
                reversed_traj_old_per_token_logps[i]
                if self.num_iterations > 1
                else per_token_logps.detach()
            )
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            unmask_positions = reversed_traj_unmask_positions[i]
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
                mean_kl += (per_token_kl * unmask_positions).sum() / unmask_positions.sum() / steps

            is_clipped = (per_token_loss1 < per_token_loss2).float()
            clip_ratio += (is_clipped * unmask_positions).sum() / unmask_positions.sum() / steps
            loss += (per_token_loss * unmask_positions).sum() / unmask_positions.sum() / steps

            # Clear step variables
            del per_token_logps, old_per_token_logps, per_token_loss, is_clipped
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )
        return loss

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        result = logits.exp() / gumbel_noise
        del noise, gumbel_noise  # Clear intermediate variables
        return result

    @torch.no_grad()
    def ass_generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_num=1,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """
        ASS generation: Exponential block-based generation with 2^i token transfer per step.
        Each block is dynamically sized by peeling off the largest 2^i from remaining tokens.
        
        Example: gen_length=127 (2^7-1), steps=7, block_num=4
        - Block 1: 15 tokens (1+2+4+8) using 4 steps
        - Block 2: 16 tokens (2^4) using 1 step  
        - Block 3: 32 tokens (2^5) using 1 step
        - Block 4: 64 tokens (2^6) using 1 step
        - + last eos token
        """
        assert gen_length >= 2 ** steps, "gen_length must be >= 2^steps"
        assert block_num <= steps, "block_num must be <= steps"

        # Calculate the power L such that 2^S - 1 = gen_length
        S = int(torch.log2(torch.tensor(gen_length, dtype=torch.float32)).item())

        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            # Calculate dynamic block structure based on block_num
            block_sizes = []
            block_steps = []
            reminder = gen_length - 2 ** S
            remaining_length = gen_length

            if block_num == 1:
                block_size = remaining_length
                remaining_length -= remaining_length
                block_step = [s for s in range(S)]
                block_sizes.insert(0, block_size)
                block_steps.insert(0, block_step)
                # Calculate the transfer tokens based on steps and block sizes
                num_transfer_tokens = self.get_num_transfer_tokens_ass(prompt, steps, block_sizes)
            else:
                # Start from the largest possible power
                current_power = S - 1  # Since gen_length = 2^S, largest single block is 2^(S-1)
                for block_idx in range(block_num - 1, -1, -1):
                    if block_idx == block_num - 1:
                        block_size = 2 ** current_power + 1 + reminder
                        remaining_length -= block_size
                        block_step = [current_power]
                        current_power -= 1
                    elif block_idx == 0:
                        # Try to distribute the sum of remaining steps into one block
                        block_size = remaining_length
                        block_step = [s for s in range(current_power + 1)]  # Use all remaining steps
                        remaining_length -= block_size
                    else:
                        block_size = 2 ** current_power
                        remaining_length -= block_size
                        block_step = [current_power]
                        current_power -= 1           
                    block_sizes.insert(0, block_size)
                    block_steps.insert(0, block_step)
                # Calculate the transfer tokens based on steps and block sizes
                num_transfer_tokens = self.get_num_transfer_tokens_ass(prompt, steps, block_sizes)

            assert remaining_length == 0, "remaining_length must be 0"
            current_pos = prompt.shape[1]
            # Process each block
            for block_idx, block_size in enumerate(block_sizes):
                end_idx = current_pos + block_size    

                block_step = block_steps[block_idx]
                for i in block_step:
                    mask_index = x == mask_id
                    # Handle classifier-free guidance
                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)

                        # Get logits in a single forward pass
                        logits = model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x).logits

                    # Apply Gumbel noise for sampling
                    logits_with_noise = self.add_gumbel_noise(logits, temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    # Handle remasking strategy
                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(dtype), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                        )
                    elif remasking == "random":
                        x0_p = torch.rand(x0.shape, device=x0.device)
                    else:
                        raise NotImplementedError(remasking)

                    # Ensure we don't process tokens beyond the current block
                    x0_p[:, end_idx:] = -np.inf

                    # Update masked tokens
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                    # Select tokens to transfer based on confidence
                    for j in range(confidence.shape[0]):
                        num_tokens = num_transfer_tokens[j, i].item()
                        if num_tokens > 0:
                            _, select_indices = torch.topk(confidence[j], k=num_tokens)
                            x[j, select_indices] = x0[j, select_indices]

                current_pos = end_idx

            return x
    
    def get_block_sizes(self, prompt, gen_length, block_num, steps):
        # Calculate dynamic block structure based on block_num
        S = int(torch.log2(torch.tensor(gen_length, dtype=torch.float32)).item())
        assert S == steps, f"logs(gen_length) {S} must be == steps {steps}"

        block_sizes = []
        block_steps = []
        reminder = gen_length - 2 ** S
        remaining_length = gen_length
            
        if block_num == 1:
            block_size = remaining_length
            remaining_length -= remaining_length
            block_step = [s for s in range(S)]
            block_sizes.insert(0, block_size)
            block_steps.insert(0, block_step)
            # Calculate the transfer tokens based on steps and block sizes
            num_transfer_tokens = self.get_num_transfer_tokens_ass(prompt, steps, block_sizes)
        else:
            # Start from the largest possible power
            current_power = S - 1  # Since gen_length = 2^S, largest single block is 2^(S-1)
            for block_idx in range(block_num - 1, -1, -1):
                if block_idx == block_num - 1:
                    block_size = 2 ** current_power + 1 + reminder
                    remaining_length -= block_size
                    block_step = [current_power]
                    current_power -= 1
                elif block_idx == 0:
                    # Try to distribute the sum of remaining steps into one block
                    block_size = remaining_length
                    block_step = [s for s in range(current_power + 1)]  # Use all remaining steps
                    remaining_length -= block_size
                else:
                    block_size = 2 ** current_power
                    remaining_length -= block_size
                    block_step = [current_power]
                    current_power -= 1           
                block_sizes.insert(0, block_size)
                block_steps.insert(0, block_step)
            # Calculate the transfer tokens based on steps and block sizes
            num_transfer_tokens = self.get_num_transfer_tokens_ass(prompt, steps, block_sizes)
        assert remaining_length == 0, "remaining_length must be 0"

        return num_transfer_tokens, block_sizes, block_steps
    
    @torch.no_grad()
    def reversed_process(
        self,
        model,
        prompt,
        steps=7,
        gen_length=128,
        block_num=1,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        assert gen_length >= 2 ** steps, f"gen_length {gen_length} must be >= 2^{steps}"
        assert block_num <= steps, f"block_num {block_num} must be <= {steps}"
        
        # Calculate the power L such that 2^S - 1 = gen_length
        S = int(torch.log2(torch.tensor(gen_length, dtype=torch.float32)).item())

        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id
            logits_to_keep = x.shape[1] - prompt.shape[1]

            num_transfer_tokens, block_sizes, block_steps = self.get_block_sizes(prompt, gen_length, block_num, steps)
            current_pos = prompt.shape[1]
            ### Record the trajectory changes of x ###
            reversed_traj = [x.clone()] # with initial x, one more than steps
            reversed_traj_unmask_positions = []
            # Process each block
            for block_idx, block_size in enumerate(block_sizes):
                end_idx = current_pos + block_size

                block_step = block_steps[block_idx]
                for i in block_step:
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id
                    with torch.cuda.amp.autocast(enabled=self.args.fp16):
                        # Handle classifier-free guidance
                        if cfg_scale > 0.0:
                            un_x = x.clone()
                            un_x[prompt_index] = mask_id
                            x_ = torch.cat([x, un_x], dim=0)
                            # Get logits in a single forward pass
                            logits = model(x_).logits
                            logits, un_logits = torch.chunk(logits, 2, dim=0)
                            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                        else:
                            logits = model(x).logits

                        # Apply Gumbel noise for sampling
                        assert temperature > 0., f"Need to introduce randomness in sampling for GRPO, got temperature > 0"
                        # Apply Gumbel noise for sampling
                        logits_with_noise = self.add_gumbel_noise(logits, temperature, dtype)
                        x0 = torch.argmax(logits_with_noise, dim=-1)
                        del logits_with_noise

                        # Handle remasking strategy
                        if remasking == "low_confidence":
                            p = F.softmax(logits.to(dtype), dim=-1)
                            x0_p = torch.squeeze(
                                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                            )
                        elif remasking == "random":
                            x0_p = torch.rand(x0.shape, device=x0.device)
                        else:
                            raise NotImplementedError(remasking)

                        # Ensure we don't process tokens beyond the current block
                        x0_p[:, end_idx:] = -np.inf

                        # Update masked tokens
                        x0 = torch.where(mask_index, x0, x)
                        confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                        # Select tokens to transfer based on confidence
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        for j in range(confidence.shape[0]):
                            num_tokens = num_transfer_tokens[j, i].item()
                            if num_tokens > 0:
                                _, select_index = torch.topk(confidence[j], k=num_tokens)
                                transfer_index[j, select_index] = True

                        x[transfer_index] = x0[transfer_index]

                        unmask_positions = (mask_index & (x != mask_id))[:, -logits_to_keep:]
                        reversed_traj.append(x.clone())
                        reversed_traj_unmask_positions.append(unmask_positions)
                        del x0, confidence, transfer_index, unmask_positions
                current_pos = end_idx

            return x, reversed_traj, reversed_traj_unmask_positions

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])

        input = batch
        logits = model(input).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)
    
    def get_num_transfer_tokens_ass(self, prompt, steps, block_sizes):
        """
        Generate exponential token transfer schedule where step i transfers 2^i tokens.
        Each block size is 2^s - 1, and total generation length is 2^L - 1.
        
        Args:
            mask_index: Boolean mask indicating which tokens are masked
            steps: Total number of diffusion steps per block
            schedule: Not used in this function but kept for compatibility
        """
        batch_size = prompt.shape[0]
        
        # Calculate number of tokens to transfer at each step: 2^i for step i
        num_transfer_tokens = torch.zeros((batch_size, steps), device=prompt.device, dtype=torch.int64)
        
        for i in range(steps):
            if i == steps - 1:
                if len(block_sizes) == 1:
                    num_transfer_tokens[:, i] = block_sizes[-1] - (2 ** i - 1)
                elif block_sizes[-1] > 2 ** i:
                    num_transfer_tokens[:, i] = block_sizes[-1]
            else:
                tokens_to_transfer = 2 ** i
                num_transfer_tokens[:, i] = tokens_to_transfer
        
        return num_transfer_tokens

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, reversed_traj, reversed_traj_unmask_positions, logits_to_keep):
        """
        Calculate per-token log probabilities.
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        steps = len(reversed_traj_unmask_positions)
        reversed_traj_logps = []

        for i in range(steps):
            prompt_ids = reversed_traj[i]
            completion_ids = reversed_traj[i+1]
            unmask_index = reversed_traj_unmask_positions[i]
            logits = self.get_logits(
                model, prompt_ids, prompt_index, self.args.cfg_scale, self.args.mask_id
            )  # [B, L, V]

            completion_logits = logits[:, -logits_to_keep:, :]
            completion_targets = completion_ids[:, -logits_to_keep:]

            per_token_logps = selective_log_softmax(
                completion_logits, completion_targets
            ).to(torch.float32)  # [B, logits_to_keep]

            del logits, completion_logits, completion_targets
            torch.cuda.empty_cache()
            # per_token_logps = torch.where(unmask_index, per_token_logps, 0)
            reversed_traj_logps.append(per_token_logps)
            # reversed_traj_logps.append(per_token_logps[unmask_index].view(batch_size, -1))
        
        return reversed_traj_logps
    
    @profiling_decorator
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        # block_length = self.args.block_length
        block_num = self.args.block_num
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            prompt_completion_ids, reversed_traj, reversed_traj_unmask_positions = self.reversed_process(
                model=unwrapped_model,
                prompt=prompt_ids,
                steps=steps,
                gen_length=gen_length,
                block_num=block_num,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=self.args.remasking,
                mask_id=self.args.mask_id,
            )

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            if self.num_iterations > 1:
                reversed_traj_old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, reversed_traj, reversed_traj_unmask_positions, logits_to_keep
                )
            else:
                reversed_traj_old_per_token_logps = None

            ### KL term ###
            if self.beta == 0.0:
                reversed_traj_ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    reversed_traj_ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, reversed_traj, reversed_traj_unmask_positions, logits_to_keep
                    )

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "reversed_traj_old_per_token_logps": reversed_traj_old_per_token_logps,
            "reversed_traj_ref_per_token_logps": reversed_traj_ref_per_token_logps,
            "advantages": advantages,
            "reversed_traj": reversed_traj,
            "reversed_traj_unmask_positions": reversed_traj_unmask_positions,
        }
