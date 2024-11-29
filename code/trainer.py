from transformers import Trainer
from tqdm import tqdm 
import gc

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import last_boxed_only_string, remove_boxed
from math_equivalence import is_equiv

class SCoRETrainer(Trainer):
    def __init__(self, config, policy_model, reference_model, train_dataset):
        """
        Initializes the SCoRETrainer with configuration, policy and reference models, and dataset.
        """

        self.config = config
        self.policy_model = policy_model
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

        epoch_pbar = tqdm(range(int(self.config['total_episodes'])), desc="Training Episodes")

        for stage in ["Stage I", "Stage II"]:

            #create optimizer and scheduler for stage I and II
            optimizer, scheduler = self.create_optimizer_and_scheduler(self.policy_model.model, 
                                                                       self.optimizer_config, self.scheduler_config)

            for episode in range(1, self.config['total_episodes'] + 1):
                
                self.policy_model.model.train()
                
                episode_reward = 0
                total_kl_div = 0
                total_first_attempt_reward = 0
                total_second_attempt_reward = 0

                for problems_batch, solutions_batch in self.get_dataloader():
                    
                    #create rollouts for batched training samples
                    first_attempt_kl_divs, first_attempt_rewards, second_attempt_kl_divs, second_attempt_rewards = self.generate_rollouts(problems_batch, solutions_batch)
                    
                    if stage == "Stage I":
                        # Stage I : Initialization
                        reward = torch.sum(- second_attempt_rewards + self.config['beta_two'] * first_attempt_kl_divs + self.config['beta_one'] * (first_attempt_kl_divs + second_attempt_kl_divs)).item()
                    else:
                        # Stage II : Reward Shaping
                        bonus_reward = self.config['alpha'] * (second_attempt_rewards - first_attempt_rewards)
                        reward = torch.sum(- bonus_reward - first_attempt_rewards + self.config['beta_one'] * (first_attempt_kl_divs + second_attempt_kl_divs)).item()

                    episode_reward += reward
                    total_kl_div += torch.sum(first_attempt_kl_divs + second_attempt_kl_divs).item()
                    total_first_attempt_reward += torch.sum(first_attempt_rewards).item()
                    total_second_attempt_reward += torch.sum(second_attempt_rewards).item()

                    optimizer.zero_grad()
                    reward.backward()
                    optimizer.step()

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
        first_messages, tokenized_first_prompts = self.prepare_first_attempt_input(
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
        first_attempt_kl_divs = self.calculate_kl_divergence(first_logits, ref_first_logits)
        
        # decode first attempt outputs
        first_decoded_completions = self.policy.tokenizer.batch_decode(first_outputs, skip_special_tokens=True) 

        # calculate first attempt rewards
        first_attempt_rewards = self.compute_rewards(first_decoded_completions, solutions_batch)

        # Cleanup first attempt variables
        del first_logits, ref_first_logits, first_outputs
        torch.cuda.empty_cache()
        
        # Second attempt template
        _, tokenized_second_prompts = self.prepare_second_attempt_input(
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
        second_attempt_kl_divs = self.calculate_kl_divergence(second_logits, ref_second_logits)

        # decode second attempt outputs
        second_decoded_completions = self.policy.tokenizer.batch_decode(second_outputs, skip_special_tokens=True)

        # calculate second attempt rewards
        second_attempt_rewards = self.compute_rewards(second_decoded_completions, solutions_batch)

        # Cleanup second attempt variables
        del second_logits, ref_second_logits, second_outputs, second_decoded_completions
        gc.collect()

        return first_attempt_kl_divs, first_attempt_rewards, second_attempt_kl_divs, second_attempt_rewards

    def calculate_kl_divergence(self, policy_logit, ref_logit):
        """
        Calculates the KL divergence between the policy model's logits and the reference model's logits.
        """

        if policy_logits.size(1) < ref_logits.size(1):
            policy_logits = F.pad(policy_logits, (0, 0, 0, ref_logits.size(1) - policy_logits.size(1)), mode='constant', value=0)
        elif ref_logits.size(1) < policy_logits.size(1):
            ref_logits = F.pad(ref_logits, (0, 0, 0, policy_logits.size(1) - ref_logits.size(1)), mode='constant', value=0)

        # Convert logits to probabilities and log probabilities
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)

        # Compute KL divergence for the batch (without reducing across batch dimension)
        kl_div = F.kl_div(policy_log_probs, ref_probs, reduction='none')  # No reduction
        kl_div_per_sample = kl_div.sum(dim=-1).mean(dim=-1).unsqueeze(1)
        return kl_div_per_sample
        
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

    def prepare_first_attempt_input(self, first_prompt, problems):
        """
        Prepares input prompts for the first attempt based on the provided problems and first attempt prompt.
        """

        first_messages = [
            [
                {"role":"system", "content": first_prompt}, 
                {"role":"user", "content": item}
            ] 
            for item in problems
        ]
        
        prompts_tokenized = self.policy_model.tokenizer.apply_chat_template(
                conversation=first_messages,            
                tools=None,                       
                add_generation_prompt=True,       
                return_dict=True,                 
                padding=True,
                truncation=True,                 
                return_tensors="pt"               
            )

        return first_messages, prompts_tokenized

    def prepare_second_attempt_input(self, first_messages, first_decoded_completions, second_prompt):
        """
        Prepares input prompts for the second attempt based on the first attempt outputs and the second attempt prompt.
        """

        second_messages = []
        for first_message, first_completion in zip(first_messages, first_decoded_completions):
            second_message = first_message.copy()
            second_message.append({"role": "assistant", "content": first_completion})
            second_message.append({"role": "user", "content": second_prompt})
            second_messages.append(second_message)
        
        prompts_tokenized = self.policy_model.tokenizer.apply_chat_template(
            conversation=second_messages,
            tools=None,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return second_messages, prompts_tokenized

    def compute_rewards(self, completions, solutions):
        """
        Computes the rewards for each completion by comparing it to the reference solution using equivalence checks.
        
        Returns a torch tensor of shape (len(completions), 1).
        """        
        rewards = []
        for completion, solution in zip(completions, solutions):

            model_answer = remove_boxed(last_boxed_only_string(completion))
            correct_answer = remove_boxed(last_boxed_only_string(solution))

            if is_equiv(model_answer, correct_answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)        
        return rewards_tensor