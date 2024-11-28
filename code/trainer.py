from transformers import Trainer 

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from torch.utils.data import DataLoader

from typing import Dict, List

from utils import last_boxed_only_string, remove_boxed, pad
from math_equivalence import is_equiv
from rolloutstorage import SCoRERolloutStorage

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
        self.rollout_store = SCoRERolloutStorage(self.policy_model.tokenizer)
    
    def train(self):
        
        model_name = 'SCoRE-'+self.config['policy_model_name']+'-Episodes-'+self.config['total_episodes']

        for episode in range(int(self.config['total_episodes'])):
            #create rollouts for all training samples
            self.generate_rollouts()
            #initiate stage one over all the rollouts              
            self.stage_one_initialization()
            #initiate stage two over all the rollouts
            self.stage_two_reward_shaping()
            #scheduler.step() for linearly decaying learning rate
            scheduler.step()

            if episode % self.config['save_interval'] == 0:
                self.model.save_pretrained(self.config['save_dir'] + model_name)


    def stage_one_initialization(self):

        dataloader = self.rollout_store.create_loader(batch_size = self.config['batch_size'], shuffle=True)
        self.policy_model.model.train()

        for batch in dataloader:
            self.optimizer.zero_grad()

            loss = - batch.second_stage_rewards.sum() + self.config['beta_two'] * batch.first_kl_divs.sum()

            loss.backward()
            self.optimizer.step()

    def stage_two_reward_shaping(self):
        
        #Second Stage with Reward Shaping
        dataloader = self.rollout_store.create_loader(batch_size = self.config['batch_size'], shuffle=True)
        self.policy_model.model.train()

        for batch in dataloader:
            self.optimizer.zero_grad()

            bonus_reward = self.config['alpha'] * (batch.second_stage_rewards - batch.first_stage_rewards)

            loss = - bonus_reward.sum() - batch.first_stage_rewards.sum() + self.config['beta_one'] * (batch.first_kl_divs.sum() + batch.second_kl_divs.sum()) 
            
            loss.backward()
            self.optimizer.step()

    def generate_rollouts(self):
        
        #clear CUDA cache
        torch.cuda.empty_cache()

        all_rollouts = []

        # Iterate over all the training samples to generate rollouts
        for problems_batch, solutions_batch in self.get_dataloader():

            print(f'problem: {problems_batch}')  
            print(f'soluchan: {solutions_batch}')  

            # First stage template
            first_messages, tokenized_first_prompts = self.prepare_first_stage_input(
                                                                                self.config['stage_one_prompt'], 
                                                                                problems_batch
                                                                                )
            print(f'first message:{first_messages}')

            # First stage policy completions
            first_outputs, first_logits = self.policy_model.generate(
                                                            tokenized_prompts['input_ids'].to(self.device), 
                                                            tokenized_prompts['attention_mask'].to(self.device),
                                                            **self.config['gen_kwargs']
                                                            )
            # First stage reference completions
            ref_first_outputs, ref_first_logits = self.reference_model.generate(
                                                            tokenized_prompts['input_ids'].to(self.device), 
                                                            tokenized_prompts['attention_mask'].to(self.device),
                                                            **self.config['gen_kwargs']
                                                            )
            
            print(f'\n\nfirst logits shape:{first_logits.shape}')
            print(f'\n\nfirst ref logits shape:{ref_first_logits.shape}')
            
            #store kl_divergence
            first_kl_div = self.calculate_kl_divergence(first_logits, solutions_batch)
            first_kl_divs.append(first_kl_div)

            print(f'\n\nfirst kl_div:{first_kl_div}')
            
            # decode first stage outputs
            first_decoded_completions = self.policy.tokenizer.batch_decode(first_outputs, skip_special_tokens=True) 

            print(f'\n\nfirst completion:{first_decoded_completions}')

            # calculate first stage rewards
            first_rewards = self.compute_rewards(first_decoded_completions, solutions)

            print(f'first rewards: {first_rewards}')
            
            # Second stage template
            second_messages, tokenized_second_prompts = self.prepare_second_stage_input(
                                                                                    first_messages, 
                                                                                    first_decoded_completions, 
                                                                                    self.config['stage_two_prompt']
                                                                                    )
            # second stage policy completions
            second_outputs, second_logits = self.policy_model.generate(
                                        tokenized_second_prompts['input_ids'].to(self.device),
                                        tokenized_second_prompts['attention_mask'].to(self.device),
                                        **self.config['gen_kwargs']
                                        )
            
            print(f'second messages: {second_messages}')

            # second stage reference completions
            ref_second_outputs, ref_second_logits = self.reference_model.generate(
                                                            tokenized_second_prompts['input_ids'].to(self.device), 
                                                            tokenized_second_prompts['attention_mask'].to(self.device),
                                                            **self.config['gen_kwargs']
                                                            )
            print(f'second logits shape: {second_logits.shape}')
            print(f'ref second logits shape: {ref_second_logits.shape}')

            # second stage kl_divergence
            
            second_kl_div = self.calculate_kl_divergence(first_logits, solutions_batch)
            second_kl_divs.append(second_kl_div)
            print(f'second kl_div: {second_kl_div}')

            # decode second stage outputs
            second_decoded_completions = self.policy.tokenizer.batch_decode(second_outputs, skip_special_tokens=True)
            print(f'second completion: {second_decoded_completions}')

            # calculate second stage rewards
            second_rewards = self.compute_rewards(first_decoded_completions, solutions_batch)
            print(f'second rewards: {second_rewards}')

            #Add all elements to the rollouts storage before pushing
            for i in range(len(problems_batch)):
                
                rollout = SCoRERLElement(
                    first_query_tensors = tokenized_first_prompts['input_ids'][i],
                    first_response_logits = first_logits[i],
                    first_response_kl_divs = first_kl_divs[i],
                    first_stage_rewards = first_rewards[i],
                    second_query_tensors = tokenized_second_prompts['input_ids'][i],
                    second_response_logits = second_logits[i],
                    second_response_kl_divs = second_kl_divs[i],
                    second_stage_rewards = second_rewards[i]
                )

                all_rollouts.append(rollout)
                print(f'Rollout {i}: {rollout}')
                sys.exit(0)

        # Add all rollouts to the rollout storage
        self.rollout_store.push(all_rollouts)
        del all_rollouts

    def calculate_kl_divergence(self, policy_logit, ref_logit):

        kl_divs = []

        for i in range(policy_logit.shape[0]):
            
            #Padding if the logit tensors are not of equal length
            if policy_logit.size(1) < ref_logit.size(1):
                policy_logit = F.pad(policy_logit, (0, 0, 0, ref_logit.size(1) - policy_logit.size(1)), mode='constant', value=0)
            elif ref_logit.size(1) < policy_logit.size(1):
                ref_logit = F.pad(ref_logit, (0, 0, 0, policy_logit.size(1) - ref_logit.size(1)), mode='constant', value=0)

            # Apply log_softmax to policy_logit and softmax to ref_logit
            log_probs1 = F.log_softmax(policy_logit, dim=-1)
            probs2 = F.softmax(ref_logit, dim=-1)
            
            # Calculate KL divergence and append to the list
            kl_div = F.kl_div(log_probs1, probs2, reduction='sum')
            kl_divs.append(kl_div)
        
        return kl_divs
        
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

    def compute_rewards(self, completions, solutions):
        
        rewards = []
        for completion, solution in zip(completions, solutions):

            model_answer = remove_boxed(last_boxed_only_string(completion))
            correct_answer = remove_boxed(last_boxed_only_string(solution))

            if is_equiv(model_answer, correct_answer):
                rewards.append(1)
            else:
                rewards.append(0)
        
        return rewards
