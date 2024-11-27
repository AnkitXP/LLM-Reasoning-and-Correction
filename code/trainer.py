from transformers import Trainer, DataCollatorWithPadding

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from torch.utils.data import DataLoader

from typing import Dict, List

from utils import last_boxed_only_string, remove_boxed
from math_equivalence import is_equiv

import sys

class SCoRETrainer(Trainer):
    def __init__(self, config, policy, ref_policy, train_dataset):

        self.config = config
        self.policy = policy
        self.ref_policy = ref_policy
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = train_dataset
        self.data_collator = DataCollatorWithPadding(self.policy.tokenizer)
    
    def train(self):
        
        model_name = 'SCoRE-'+config['policy_model_name']+'-Episodes-'+config['total_episodes']
        for episode in config['total_episodes']:
            self.stage_one_initialization()
            self.stage_two_reward_shaping()
        
        self.model.save_pretrained(config['save_dir'] + model_name)

    def stage_one_initialization(self):

        stage_one_prompt = self.config['stage_one_prompt']

        for batch in self.get_dataloader():            
            with self.optimizer.zero_grad():
                
                batch = self.prepare_first_stage_input(stage_one_prompt, batch)

                # First attempt
                first_outputs = self.policy(**batch.to(self.device), **self.config['gen_kwargs'])
                base_outputs = self.ref_policy(**batch.to(self.device), **self.config['gen_kwargs'])

                print(first_outputs)
                print(base_outputs)

                sys.exit(0)
                
                # KL divergence loss
                kl_loss = nn.KLDivLoss(reduction="batchmean")(
                    nn.functional.log_softmax(first_outputs.logits, dim=-1),
                    nn.functional.softmax(base_outputs.logits, dim=-1)
                )
                
                # Second attempt
                second_inputs = self.prepare_second_attempt_input(batch, first_outputs)
                second_outputs = self.model(**second_inputs)
                
                # Reward for second attempt
                reward = self.compute_reward(second_outputs, batch['labels'])
            break
            loss = -reward + self.config['kl_weight'] * kl_loss
            loss.backward()
            self.optimizer.step()

    def stage_two_reward_shaping(self):
        dataloader = self.dataloader
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # First attempt
            first_outputs = self.model(**batch)
            first_reward = self.compute_reward(first_outputs, batch['labels'])
            
            # Second attempt
            second_inputs = self.prepare_second_attempt_input(batch, first_outputs)
            second_outputs = self.model(**second_inputs)
            second_reward = self.compute_reward(second_outputs, batch['labels'])
            
            # Shaped reward
            shaped_reward = second_reward + self.config['progress_weight'] * (second_reward - first_reward)
            
            loss = -shaped_reward
            loss.backward()
            self.optimizer.step()
    
    def get_dataloader(self):
        """
        Creates and returns a DataLoader for the training dataset.

        Returns:
            DataLoader: Configured DataLoader instance for training.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],  # Batch size from config
            shuffle=True,  # Shuffle the data for training
            collate_fn=self.data_collator,  # Use the data collator for padding and truncation
            drop_last=True  # Drop the last batch if it's smaller than batch_size
        )

    def reward_function(self, model_answer, problem_solution):
        model_answer = remove_boxed(last_boxed_only_string(model_output))
        correct_answer = remove_boxed(last_boxed_only_string(problem_solution))
        return is_equiv(model_answer, correct_answer) 

    def generate_completions(self):
        # Implementation for generating completions
        pass

    def prepare_second_stage_input(self, batch, first_outputs):
        # Implementation for preparing input for second attempt
        pass

    def prepare_first_stage_input(self, prompt, batch):
        # Implementation for preparing input for first attempt

        # Tokenize the prompt once
        prompt_tokens = self.policy.tokenizer.encode_plus(
            prompt,
            return_tensors='pt',
            padding=False,
            truncation=False  # Prompts are typically short
        )

        prompt_input_ids = prompt_tokens['input_ids'].squeeze(0)
        prompt_attention_mask = prompt_tokens['attention_mask'].squeeze(0)
            
        # Add the prompt tokens to each problem in the batch
        batch['input_ids'] = torch.cat([
            prompt_input_ids.repeat(batch['problem_input_ids'].size(0), 1),  # Repeat prompt for each sample
            batch['problem_input_ids']
        ], dim=1)  # Concatenate along the sequence dimension

        batch['attention_mask'] = torch.cat([
            prompt_attention_mask.repeat(batch['problem_attention_mask'].size(0), 1),  # Repeat mask for each sample
            batch['problem_attention_mask']
        ], dim=1)
        
        return batch

    def compute_reward(self, outputs, labels):
        # Implementation for computing reward
        pass