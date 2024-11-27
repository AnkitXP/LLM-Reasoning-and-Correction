import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import config

class PolicyModel():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(config['policy_model_name']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config['policy_model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, inputs, **gen_kwargs):
        outputs = self.model.generate(**inputs, **gen_kwargs)
        return outputs
    
    def save_model(self, save_dir, model_name='policy_model'):
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, model_name)

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")