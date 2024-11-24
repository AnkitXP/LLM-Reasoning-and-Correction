import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import config

class PolicyModel():
    def __init__(self, trainable='True'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(config['policy_model_name']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config['policy_model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if trainable:
            self.model.train()
        else:
            self.model.eval()

    def generate(self, inputs, **gen_kwargs):

        if isinstance(inputs, str):
            inputs = self.tokenizer(inputs, return_tensors='pt').input_ids.to(self.device)

        outputs = self.model.generate(inputs, **gen_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def save_model(self, save_dir, model_name='policy_model'):
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, model_name)

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_dir):
        
        print(f"Loading model from: {load_dir}")
        self.model = AutoModelForCausalLM.from_pretrained(load_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)