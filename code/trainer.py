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

from utils import check_correct, truncate_response, first_true_indices

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

                    reward = reward.sum()

                    episode_reward += reward.item()
                    total_kl_div += torch.sum(first_attempt_kl_divs + second_attempt_kl_divs).item()
                    total_first_attempt_reward += torch.sum(first_attempt_rewards).item()
                    total_second_attempt_reward += torch.sum(second_attempt_rewards).item()

                    del first_attempt_kl_divs, first_attempt_rewards, second_attempt_kl_divs, second_attempt_rewards
                    gc.collect()
                    torch.cuda.empty_cache()

                    reward.requires_grad_(True)
                    reward = reward.to(self.policy_model.device)
                    reward = reward / accumulation_steps
                    reward.backward()

                    # Update progress bar after every batch
                    epoch_pbar.update(1)

                    # Accumulate gradients and perform optimization step
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.get_dataloader()):
                        optimizer.step()
                        optimizer.zero_grad()

                # Statistics for Log
                avg_kl_div = total_kl_div / len(self.get_dataloader())
                avg_first_attempt_reward = total_first_attempt_reward / len(self.get_dataloader())
                avg_second_attempt_reward = total_second_attempt_reward / len(self.get_dataloader())

                # Update progress bar description
                epoch_pbar.set_description(f"{stage} - Episode {episode}/{self.config['total_episodes']} - Reward: {episode_reward:.4f}")

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
        
        #clear CUDA cache
        torch.cuda.empty_cache()

        # First attempt template
        first_messages, tokenized_first_prompts = self.policy_model.prepare_first_attempt_input(
                                                                            self.config['first_attempt_prompt'], 
                                                                            problems_batch
                                                                            )

        # First attempt policy completions
        first_outputs, first_logits = self.policy_model.generate(
                                                        tokenized_first_prompts['input_ids'].to(self.policy_model.device), 
                                                        tokenized_first_prompts['attention_mask'].to(self.policy_model.device),
                                                        **self.config['gen_kwargs']
                                                        )
        # First attempt reference completions
        _, ref_first_logits = self.reference_model.generate(
                                                        tokenized_first_prompts['input_ids'].to(self.reference_model.device), 
                                                        tokenized_first_prompts['attention_mask'].to(self.reference_model.device),
                                                        **self.config['gen_kwargs']
                                                        )
                    
        #store kl_divergence
        first_attempt_kl_divs = self.calculate_kl_divergence(first_logits, ref_first_logits, first_outputs)
        
        # decode first attempt outputs
        first_decoded_completions = self.policy_model.tokenizer.batch_decode(first_outputs, skip_special_tokens=True) 

        # calculate first attempt rewards
        first_attempt_rewards = self.compute_rewards(first_decoded_completions, solutions_batch)

        # Cleanup first attempt variables
        del first_logits, ref_first_logits, first_outputs
        torch.cuda.empty_cache()
        
        # Second attempt template
        _, tokenized_second_prompts = self.policy_model.prepare_second_attempt_input(
                                                                    first_messages, 
                                                                    first_decoded_completions, 
                                                                    self.config['second_attempt_prompt']
                                                                    )
        # second attempt policy completions
        second_outputs, second_logits = self.policy_model.generate(
                                    tokenized_second_prompts['input_ids'].to(self.policy_model.device),
                                    tokenized_second_prompts['attention_mask'].to(self.policy_model.device),
                                    **self.config['gen_kwargs']
                                    )
        
        # second attempt reference completions
        _, ref_second_logits = self.reference_model.generate(
                                                        tokenized_second_prompts['input_ids'].to(self.reference_model.device), 
                                                        tokenized_second_prompts['attention_mask'].to(self.reference_model.device),
                                                        **self.config['gen_kwargs']
                                                        )

        del tokenized_second_prompts, first_messages, first_decoded_completions

        # second attempt kl_divergence   
        second_attempt_kl_divs = self.calculate_kl_divergence(second_logits, ref_second_logits, second_outputs)

        # decode second attempt outputs
        second_decoded_completions = self.policy_model.tokenizer.batch_decode(second_outputs, skip_special_tokens=True)

        # calculate second attempt rewards
        second_attempt_rewards = self.compute_rewards(second_decoded_completions, solutions_batch)

        # Cleanup second attempt variables
        del second_logits, ref_second_logits, second_outputs, second_decoded_completions
        gc.collect()

        return first_attempt_kl_divs, first_attempt_rewards, second_attempt_kl_divs, second_attempt_rewards

    def calculate_kl_divergence(self, policy_logits, ref_logits, response):
        """
        Calculates the KL divergence between the policy model's logits and the reference model's logits.
        """

        INVALID_LOGPROB = 1.0  # Set to -inf for invalid logprobs (padding tokens)

        # Compute log probabilities for policy model
        all_logprob = F.log_softmax(policy_logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        policy_logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)

        # Temperature scaling for the reference logits
        ref_logits /= self.config['gen_kwargs']['temperature'] + 1e-7
        ref_all_logprob = F.log_softmax(ref_logits, dim=-1)  # (batch_size, seq_len, vocab_size)
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

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, requires_grad = True).unsqueeze(1)        
        return rewards_tensor.to(self.policy_model.device)
