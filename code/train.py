import os
import torch
from dataset import MATH
from model import PolicyModel
from trainer import SCoRETrainer

from config import config  # Import configurations

def train_model():

    #Instantiate Policy and Reference Models
    policy_model = PolicyModel()
    ref_model = PolicyModel()

    #Create Dataset
    train_dataset = MATH()

    #Instantiate trainer and initiate training
    trainer = SCoRETrainer(config, policy_model, ref_model, train_dataset)
    trainer.train()

    # #============================================================================================

    # # Select a batch of problems
    # batch_size = 5  # You can adjust this based on your needs
    # batch_problems = [train_dataset[i]['problem'] for i in range(batch_size)]


    # """
    # List with manual append prompt
    # """
    # # prompts = [config['stage_one_prompt'] + problem for problem in batch_problems]
    # # prompts_tokenized = policy_model.tokenizer(prompts, max_length=500, padding=True, truncation=True, return_tensors='pt')


    # """
    # Chat Template
    # """
    # prompt = config['stage_one_prompt']
    # messages = [
    #     [
    #         {"role":"system", "content": prompt}, 
    #         {"role":"user", "content": item}
    #     ] 
    #     for item in batch_problems
    # ]
    
    # prompts_tokenized = policy_model.tokenizer.apply_chat_template(
    #         conversation=messages,            
    #         tools=None,                       
    #         add_generation_prompt=True,       
    #         return_dict=True,                 
    #         padding=True,
    #         truncation=True,                 
    #         return_tensors="pt"               
    #     )

    # # Generate outputs for the batch
    # outputs, logits = policy_model.generate(
    #     input_ids=prompts_tokenized['input_ids'].to(policy_model.device),
    #     attention_mask=prompts_tokenized['attention_mask'].to(policy_model.device),
    #     **config['gen_kwargs']
    # )

    # decoded_texts = [policy_model.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]

    # for i, text in enumerate(decoded_texts):
    #     print(f"\n\nGenerated text {i + 1}: {text}")