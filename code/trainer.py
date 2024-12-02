import transformers
transformers.logging.set_verbosity_error()

from transformers import Trainer
from tqdm import tqdm 
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import check_correct, truncate_response, first_true_indices, forward

class SCoRETrainer(Trainer):
    def __init__(self, config, policy_model, reference_model, train_dataset):
        """
        Initializes the SCoRETrainer with configuration, policy and reference models, and dataset.
        """
        self.config = config

        policy_model.model.requires_grad_(True)
        self.policy_model = policy_model

        reference_model.model.requires_grad_(False)
        reference_model.model = reference_model.model.eval()
        self.reference_model = reference_model

        self.dataset = train_dataset
        self.writer = SummaryWriter(log_dir=self.config['log_dir'])
        self.optimizer_config = {
                            "lr": self.config['lr'], 
                            "betas": (0.9, 0.999), 
                            "eps": self.config['total_episodes'], 
                            "weight_decay": 0.01
                            }
        self.scheduler_config = {
                            "lr_lambda": lambda epoch: 0.95 ** epoch
                            }
    
    def train(self):
        """
        Executes the training process, including rollouts, stage one initialization, and stage two reward shaping.
        """

        accumulation_steps = self.config['gradient_accumulation_steps']
        total_batches = len(self.get_dataloader()) * self.config['total_episodes']
        accumulated_batch_size = self.config['batch_size'] * accumulation_steps
        epoch_pbar = tqdm(total=total_batches, desc="Training Progress")

        for stage in ["Stage I", "Stage II"]:

            #create optimizer and scheduler for stage I and II
            optimizer, scheduler = self.create_optimizer_and_scheduler(
                                                                    self.policy_model.model, 
                                                                    self.optimizer_config, 
                                                                    self.scheduler_config
                                                                    )

            for episode in range(1, self.config['total_episodes'] + 1):
                
                self.policy_model.model.train()
                
                episode_reward = 0
                total_kl_div = 0
                total_first_attempt_reward = 0
                total_second_attempt_reward = 0

                for batch_idx, (problems_batch, solutions_batch) in enumerate(self.get_dataloader()):

                    optimizer.zero_grad()
                        
                    #create rollouts for batched training samples
                    first_attempt_kl_divs, first_attempt_rewards, second_attempt_kl_divs, second_attempt_rewards = self.generate_rollouts(problems_batch, solutions_batch)

                    if stage == "Stage I":
                        # Stage I : Initialization
                        reward = torch.sum(
                            - second_attempt_rewards 
                            + self.config['beta_two'] * first_attempt_kl_divs 
                            + self.config['beta_one'] * (first_attempt_kl_divs + second_attempt_kl_divs)
                            )
                    else:
                        # Stage II : Reward Shaping
                        bonus_reward = self.config['alpha'] * (second_attempt_rewards - first_attempt_rewards)
                        reward = torch.sum(
                            - bonus_reward 
                            - first_attempt_rewards 
                            + self.config['beta_one'] * (first_attempt_kl_divs + second_attempt_kl_divs)
                            )

                    episode_reward += reward.item()
                    total_kl_div += torch.sum(first_attempt_kl_divs + second_attempt_kl_divs).item()
                    total_first_attempt_reward += torch.sum(first_attempt_rewards).item()
                    total_second_attempt_reward += torch.sum(second_attempt_rewards).item()

                    del first_attempt_kl_divs, first_attempt_rewards, second_attempt_kl_divs, second_attempt_rewards
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    gc.collect()

                    reward.requires_grad_(True)
                    reward = reward.to(self.policy_model.device)
                    reward = reward / accumulation_steps
                    reward.backward()

                    # Update progress bar with current batch reward
                    epoch_pbar.set_postfix({
                        "Stage": stage,
                        "Episode": f"{episode}/{self.config['total_episodes']}",
                        "Batch Reward": reward.item()
                    })
                    epoch_pbar.update(1)

                    # Accumulate gradients and perform optimization step
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.get_dataloader()):
                        optimizer.step()
                        optimizer.zero_grad()

                # Statistics for Log
                avg_kl_div = total_kl_div / len(self.get_dataloader())
                avg_first_attempt_reward = total_first_attempt_reward / len(self.get_dataloader())
                avg_second_attempt_reward = total_second_attempt_reward / len(self.get_dataloader())

                print(f'Episode {episode} {stage} completed. Total rewards: {episode_reward:.4f}')

                # Log metrics to TensorBoard
                self.writer.add_scalar(f'{stage}/Reward', episode_reward, episode)
                self.writer.add_scalar(f'{stage}/Avg_KL_Divergence', avg_kl_div, episode)
                self.writer.add_scalar(f'{stage}/Avg_First_Attempt_Reward', avg_first_attempt_reward, episode)
                self.writer.add_scalar(f'{stage}/Avg_Second_Attempt_Reward', avg_second_attempt_reward, episode)
                self.writer.add_scalar(f'{stage}/Learning_Rate', scheduler.get_last_lr()[0], episode)
                
                # Decay Learning Rate after every episode
                scheduler.step()

        # Close TensorBoard writer
        self.writer.close()

        #Save Model
        self.policy_model.save_model()

    def create_optimizer_and_scheduler(self, model, optimizer_config, scheduler_config):
        """
        Creates the optimizer and scheduler
        """
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, **scheduler_config)
        return optimizer, scheduler

    def generate_rollouts(self, problems_batch, solutions_batch):
        """
        Generates rollouts for processing the training samples through the first and second attempts.

        Returns: Tensors for first_attempt_kl_divs, first_attempt_rewards, second_attempt_kl_divs, second_attempt_rewards
        """

        # Initialize storage for results
        all_first_attempt_kl_divs = []
        all_first_attempt_rewards = []
        all_second_attempt_kl_divs = []
        all_second_attempt_rewards = []

        # Get batch size for rollouts
        batch_size = self.config['local_rollout_forward_batch_size']

        for i in range(0, len(problems_batch), batch_size):
            # Slice mini-batches
            problems_mini_batch = problems_batch[i : i + batch_size]
            solutions_mini_batch = solutions_batch[i : i + batch_size]

            # FIRST ATTEMPT

            # Prepare first attempt inputs
            first_messages, tokenized_first_prompts = self.policy_model.prepare_first_attempt_input(
                self.config['first_attempt_prompt'], problems_mini_batch
            )
            first_attempt_context_length = tokenized_first_prompts['input_ids'].shape[1]

            # Generate first attempt policy completions
            first_outputs, first_logits = self.policy_model.generate(
                input_ids=tokenized_first_prompts['input_ids'], **self.config['gen_kwargs']
            )
            first_attempt_generations = first_outputs[:, first_attempt_context_length:]

            # Generate first attempt reference logits
            ref_first_output = forward(self.reference_model.model, first_outputs, self.reference_model.tokenizer.pad_token_id)
            ref_first_logits = ref_first_output.logits[:, first_attempt_context_length - 1 : -1]

            # Calculate first attempt KL divergence
            first_attempt_kl_divs = self.calculate_kl_divergence(
                first_logits, ref_first_logits, first_attempt_generations
            )

            # Decode first attempt outputs
            first_decoded_completions = self.policy_model.tokenizer.batch_decode(
                first_attempt_generations, skip_special_tokens=True
            )

            # Calculate first attempt rewards
            first_attempt_rewards = self.compute_rewards(first_decoded_completions, solutions_mini_batch)

            del(
                tokenized_first_prompts,
                first_logits,
                first_attempt_generations,
                ref_first_output,
                ref_first_logits
            )

            # SECOND ATTEMPT

            # Prepare second attempt inputs
            _, tokenized_second_prompts = self.policy_model.prepare_second_attempt_input(
                first_messages, first_decoded_completions, self.config['second_attempt_prompt']
            )

            del first_messages, first_decoded_completions

            second_attempt_context_length = tokenized_second_prompts['input_ids'].shape[1]

            # Generate second attempt policy completions
            second_outputs, second_logits = self.policy_model.generate(
                input_ids=tokenized_second_prompts['input_ids'], **self.config['gen_kwargs']
            )
            second_attempt_generations = second_outputs[:, second_attempt_context_length:]

            # Generate second attempt reference logits
            ref_second_output = forward(
                self.reference_model.model, second_outputs, self.reference_model.tokenizer.pad_token_id
            )
            ref_second_logits = ref_second_output.logits[:, second_attempt_context_length - 1 : -1]

            # Calculate second attempt KL divergence
            second_attempt_kl_divs = self.calculate_kl_divergence(
                second_logits, ref_second_logits, second_attempt_generations
            )

            # Decode second attempt outputs
            second_decoded_completions = self.policy_model.tokenizer.batch_decode(
                second_attempt_generations, skip_special_tokens=True
            )

            # Calculate second attempt rewards
            second_attempt_rewards = self.compute_rewards(second_decoded_completions, solutions_mini_batch)

            # Collect results from mini-batch
            all_first_attempt_kl_divs.append(first_attempt_kl_divs)
            all_first_attempt_rewards.append(first_attempt_rewards)
            all_second_attempt_kl_divs.append(second_attempt_kl_divs)
            all_second_attempt_rewards.append(second_attempt_rewards)

            # Cleanup for this mini-batch
            del(
                tokenized_second_prompts,
                second_outputs,
                second_logits,
                ref_second_output,
                ref_second_logits,
                second_decoded_completions,
                first_attempt_kl_divs,
                first_attempt_rewards,
                second_attempt_kl_divs,
                second_attempt_rewards
            )

        # Concatenate results from all mini-batches
        first_attempt_kl_divs = torch.cat(all_first_attempt_kl_divs, dim=0)
        first_attempt_rewards = torch.cat(all_first_attempt_rewards, dim=0)
        second_attempt_kl_divs = torch.cat(all_second_attempt_kl_divs, dim=0)
        second_attempt_rewards = torch.cat(all_second_attempt_rewards, dim=0)

        return first_attempt_kl_divs, first_attempt_rewards, second_attempt_kl_divs, second_attempt_rewards
        

    def calculate_kl_divergence(self, policy_logits, ref_logits, response):
        """
        Calculates the KL divergence between the policy model's logits and the reference model's logits.
        """

        INVALID_LOGPROB = 1.0  # Set to -inf for invalid logprobs (padding tokens)

        # Compute log probabilities for policy model
        all_logprob = F.log_softmax(policy_logits, dim=-1, dtype=torch.bfloat16)  # (batch_size, seq_len, vocab_size)
        policy_logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)

        # Temperature scaling for the reference logits
        ref_logits /= self.config['gen_kwargs']['temperature'] + 1e-7
        ref_all_logprob = F.log_softmax(ref_logits, dim=-1, dtype=torch.bfloat16)  # (batch_size, seq_len, vocab_size)
        ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)

        # Truncate response (e.g., if it reaches an EOS token)
        postprocessed_response = response
        postprocessed_response = truncate_response(
            self.policy_model.tokenizer.eos_token_id, 
            self.policy_model.tokenizer.pad_token_id, 
            response
        )

        # Get the sequence length (valid tokens)
        sequence_length = first_true_indices(postprocessed_response == self.policy_model.tokenizer.pad_token_id) - 1

        # Create a padding mask for valid tokens
        response_idxs = torch.arange(response.shape[1], device=self.policy_model.device).repeat(response.shape[0], 1)
        padding_mask = response_idxs >= sequence_length.unsqueeze(1)  # Mask out padding tokens (where idx >= seq_length)

        # Mask out padding tokens in the log probabilities (set to -inf for invalid tokens)
        policy_logprob = torch.masked_fill(policy_logprob, padding_mask, INVALID_LOGPROB)
        ref_logprob = torch.masked_fill(ref_logprob, padding_mask, INVALID_LOGPROB)

        # Compute KL divergence: log(P) - log(Q)
        kl_div = policy_logprob - ref_logprob

        # Sum the KL divergence over the sequence and normalize by the number of valid tokens
        kl_div_per_sample = kl_div.sum(dim=1) / sequence_length.clamp(min=1e-9)  # Avoid division by zero

        return kl_div_per_sample.unsqueeze(1)

        
    def get_dataloader(self):
        """
        Creates and returns a DataLoader for the training dataset.
        """

        return DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],  
            shuffle=True,
            drop_last=True
        )

    def compute_rewards(self, completions, solutions):
        """
        Computes the rewards for each completion by comparing it to the reference solution using equivalence checks.

        Returns a torch tensor of shape (len(completions), 1).
        """        
        rewards = check_correct(completions, solutions)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float16, requires_grad = True)       
        return rewards_tensor.to(self.policy_model.device)
