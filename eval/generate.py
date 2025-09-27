import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
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

def get_num_transfer_tokens_ass(prompt, steps, block_sizes):
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

def sampling(logits, x0, remasking, eos_id, dtype, step_idx=None, total_steps=None, eos_min_gamma=0.4, eos_max_gamma=1.0):
        assert remasking == "low_confidence", f"Expected remasking == low_confidence in sampling"
        p = F.softmax(logits.to(dtype), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )
        # Soft, step-dependent EOS suppression
        min_fac = eos_min_gamma 
        max_fac = eos_max_gamma
        if step_idx is not None and total_steps is not None and total_steps > 1:
            t = min(max(step_idx, 0), total_steps - 1) / (total_steps - 1)
            fac = min_fac + (max_fac - min_fac) * t  # linear schedule: strong suppression early -> none late
        else:
            fac = max_fac
        x0_p = torch.where(x0 == eos_id, x0_p * fac, x0_p)
        return x0_p

def ass_sampling(logits, x0, remasking, eos_id, dtype, step_idx=None, total_steps=None, eos_min_gamma=0.01, eos_max_gamma=1.0):
        assert remasking == "low_confidence", f"Expected remasking == low_confidence in sampling"
        p = F.softmax(logits.to(dtype), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )
        # Soft, step-dependent EOS suppression
        min_fac = eos_min_gamma 
        max_fac = eos_max_gamma
        if step_idx is not None and total_steps is not None and total_steps > 1:
            t = (2 ** (step_idx+1) - 1) / 2 ** total_steps
            fac = min_fac + (max_fac - min_fac) * t  # linear schedule: strong suppression early -> none late
        else:
            fac = max_fac
        x0_p = torch.where(x0 == eos_id, x0_p * fac, x0_p)
        return x0_p

@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Optimized version of the generate function.
    """
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
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
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
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
        return x

@torch.no_grad()
def eoser_generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_min_gamma=0.4,
    eos_max_gamma=1.0,
):
    """
    Optimized version of the generate function.
    """
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
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
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    dtype = model.dtype
                    x0_p = sampling(
                                logits=logits,
                                x0=x0,
                                remasking=remasking,
                                eos_id=tokenizer.eos_token_id,
                                dtype=dtype,
                                step_idx=i,
                                total_steps=steps,
                                eos_min_gamma=eos_min_gamma,
                                eos_max_gamma=eos_max_gamma,
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
        return x
    
@torch.no_grad()
def ass_generate(
    model,
    prompt,
    tokenizer,
    steps=7,  # Total steps for gen_length=127 (2^7-1) + 1, the last token == eos token
    gen_length=128,  # Should be 2^S
    block_num=4,  # Number of blocks to split generation into
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    BASS generation: Exponential block-based generation with 2^i token transfer per step.
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
    
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
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
            num_transfer_tokens = get_num_transfer_tokens_ass(prompt, steps, block_sizes)
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
            num_transfer_tokens = get_num_transfer_tokens_ass(prompt, steps, block_sizes)

        assert remaining_length == 0, "remaining_length must be 0"
        current_pos = prompt.shape[1]
        # Process each block
        for block_idx, block_size in tqdm(enumerate(block_sizes), disable=(dist.get_rank() != 0)):
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
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
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

@torch.no_grad()
def eoser_ass_generate(
    model,
    prompt,
    tokenizer,
    steps=7,  # Total steps for gen_length=127 (2^7-1) + 1, the last token == eos token
    gen_length=128,  # Should be 2^S
    block_num=4,  # Number of blocks to split generation into
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_min_gamma=0.01,
    eos_max_gamma=1.0,
):
    """
    BASS generation: Exponential block-based generation with 2^i token transfer per step.
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
    
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
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
            num_transfer_tokens = get_num_transfer_tokens_ass(prompt, steps, block_sizes)
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
            num_transfer_tokens = get_num_transfer_tokens_ass(prompt, steps, block_sizes)

        assert remaining_length == 0, "remaining_length must be 0"
        current_pos = prompt.shape[1]
        # Process each block
        for block_idx, block_size in tqdm(enumerate(block_sizes), disable=(dist.get_rank() != 0)):
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
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    dtype = model.dtype
                    x0_p = ass_sampling(
                            logits=logits,
                            x0=x0,
                            remasking=remasking,
                            eos_id=tokenizer.eos_token_id,
                            dtype=dtype,
                            step_idx=i,
                            total_steps=steps,
                            eos_min_gamma=eos_min_gamma,
                            eos_max_gamma=eos_max_gamma,
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
