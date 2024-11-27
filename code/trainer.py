from transformers import Trainer 

from torch.optim import AdamW, LambdaLR

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from torch.utils.data import DataLoader

from typing import Dict, List

from utils import last_boxed_only_string, remove_boxed, pad
from math_equivalence import is_equiv

import sys

class SCoRETrainer(Trainer):
    def __init__(self, config, policy_model, reference_model, train_dataset):

        self.config = config
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = train_dataset
        self.optimizer = AdamW(
                                self.policy_model.model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=0.01  # Recommended for large-scale models
                              )
        self.scheduler = LambdaLR(self.optimizer)
    
    def train(self):
        
        model_name = 'SCoRE-' + self.config['policy_model_name']
        
        for episode in range(int(self.config['total_episodes'])):

            torch.cuda.empty_cache()
            
            self.stage_one_initialization()
            self.stage_two_reward_shaping()

            if episode%config['save_interval'] == 0:
                self.policy_model.save_pretrained(self.config['save_dir'] + model_name +'-Episode-'+self.config['episode'])
        
        self.model.save_pretrained(self.config['save_dir'] + model_name +'-Final-Episode')

    def stage_one_initialization(self):

        rewards = []
        kl_divs = []

        self.policy_model.model.train()

        for problems_batch, solutions_batch in self.get_dataloader():  

            with self.optimizer.zero_grad():
                
                first_messages, tokenized_first_prompts = self.prepare_first_stage_input(
                                                                                    self.config['stage_one_prompt'], 
                                                                                    problems_batch
                                                                                    )

                # First attempt completions
                first_outputs, first_logits = self.policy_model.generate(
                                                                tokenized_prompts['input_ids'].to(self.device), 
                                                                tokenized_prompts['attention_mask'].to(self.device),
                                                                **self.config['gen_kwargs']
                                                                )

                ref_outputs, ref_logits = self.reference_model.generate(
                                                                tokenized_prompts['input_ids'].to(self.device), 
                                                                tokenized_prompts['attention_mask'].to(self.device),
                                                                **self.config['gen_kwargs']
                                                                )
                
                #store batch mean kl_divergence
                kl_divs.append(self.calculate_kl_divergence(first_logits, ref_logits))

                first_decoded_completions = self.policy.tokenizer.batch_decode(first_outputs, skip_special_tokens=True) 
                
                # Second attempt completions
                second_messages, tokenized_second_prompts = self.prepare_second_stage_input(
                                                                                        first_messages, 
                                                                                        first_decoded_completions, 
                                                                                        self.config['stage_two_prompt']
                                                                                        )
                
                second_outputs, second_logits = self.policy_model.generate(
                                         tokenized_second_prompts['input_ids'].to(self.device),
                                         tokenized_second_prompts['attention_mask'].to(self.device),
                                         **self.config['gen_kwargs']
                                            )
                
                second_decoded_completions = self.policy.tokenizer.batch_decode(second_outputs, skip_special_tokens=True)

                # Reward for second attempt
                rewards.append(self.compute_reward(second_decoded_completions, solutions_batch).mean())
            
            loss = - rewards.sum() + self.config['beta_two'] * kl_divs.sum()
            loss.backward()
            self.optimizer.step()

    def stage_two_reward_shaping(self):

        self.policy_model.model.train()

        for problems_batch, solutions_batch in self.get_dataloader():
            with self.optimizer.zero_grad():
                first_messages, tokenized_first_prompts = self.prepare_first_stage_input(
                    self.config['stage_one_prompt'],
                    problems_batch
                )

                # First attempt completions
                first_outputs, first_logits = self.policy_model.generate(
                    tokenized_first_prompts['input_ids'].to(self.device),
                    tokenized_first_prompts['attention_mask'].to(self.device),
                    **self.config['gen_kwargs']
                )

                ref_outputs, ref_logits = self.reference_model.generate(
                                                                tokenized_prompts['input_ids'].to(self.device), 
                                                                tokenized_prompts['attention_mask'].to(self.device),
                                                                **self.config['gen_kwargs']
                                                                )

                first_decoded_completions = self.policy_model.tokenizer.batch_decode(first_outputs, skip_special_tokens=True)
                first_rewards = self.compute_reward(first_decoded_completions, solutions_batch)

                # Second attempt completions
                second_messages, tokenized_second_prompts = self.prepare_second_stage_input(
                    first_messages,
                    first_decoded_completions,
                    self.config['stage_two_prompt']
                )

                second_outputs, second_logits = self.policy_model.generate(
                    tokenized_second_prompts['input_ids'].to(self.device),
                    tokenized_second_prompts['attention_mask'].to(self.device),
                    **self.config['gen_kwargs']
                )

                second_decoded_completions = self.policy_model.tokenizer.batch_decode(second_outputs, skip_special_tokens=True)
                second_rewards = self.compute_reward(second_decoded_completions, solutions_batch)

                # Compute shaped reward
                shaped_rewards = second_rewards + self.config['alpha'] * (second_rewards - first_rewards)
                rewards.append(shaped_rewards.mean())

                # Compute KL divergence
                kl_div = self.calculate_kl_divergence(second_logits, first_logits)
                kl_divs.append(kl_div)

            # Compute loss and update
            loss = -rewards[-1] + self.config['beta_two'] * kl_divs[-1]
            loss.backward()
            self.optimizer.step()

        return rewards, kl_divs

    def calculate_kl_divergence(self, policy_logit, ref_logit):
        
        if first_logits.size(2) < base_logits.size(2):
            first_logits = F.pad(first_logits, (0, base_logits.size(2) - first_logits.size(2)), mode='constant', value=0)
        elif base_logits.size(2) < first_logits.size(2):
            base_logits = F.pad(base_logits, (0, first_logits.size(2) - base_logits.size(2)), mode='constant', value=0)

        # Apply log_softmax to first_logits and softmax to base_logits
        log_probs1 = F.log_softmax(first_logits, dim=-1)
        probs2 = F.softmax(base_logits, dim=-1)
        
        # Calculate KL divergence
        kl_div = F.kl_div(log_probs1, probs2, reduction='batchmean')
        
        return kl_div
        
    def get_dataloader(self):
        """
        Creates and returns a DataLoader for the training dataset.

        Returns:
            DataLoader: Configured DataLoader instance for training.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],  
            shuffle=True,
            drop_last=True
        )

    def prepare_first_stage_input(self, first_prompt, problems):
        # Implementation for preparing input for first attempt
        first_messages = [
            [
                {"role":"system", "content": first_prompt}, 
                {"role":"user", "content": item}
            ] 
            for item in problems
        ]
        
        prompts_tokenized = policy_model.tokenizer.apply_chat_template(
                conversation=first_messages,            
                tools=None,                       
                add_generation_prompt=True,       
                return_dict=True,                 
                padding=True,
                truncation=True,                 
                return_tensors="pt"               
            )

        return first_messages, prompts_tokenized

    def prepare_second_stage_input(self, first_messages, first_decoded_completions, second_prompt):
        
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

    def compute_reward(self, completions, solutions):
        
        rewards = []
        for completion, solution in zip(completions, solutions):

            model_answer = remove_boxed(last_boxed_only_string(completion))
            correct_answer = remove_boxed(last_boxed_only_string(solution))

            if is_equiv(model_answer, correct_answer):
                rewards.append(1)
            else:
                rewards.append(0)
        
        return rewards
