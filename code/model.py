import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import config

class PolicyModel():
    def __init__(self):
        """
        Initialize the policy model and tokenizer
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(config['policy_model_name']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config['policy_model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_ids, attention_mask=None, **gen_kwargs):
        """
        Generates only the completion and respective logits based on input length
        """
        outputs = self.model.generate(  
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        **gen_kwargs
                        )

        # Extract only the completion part
        input_length = input_ids.shape[1]
        completions = outputs.sequences[:, input_length:]
        
        # Stack the logits for the completion part only
        logits = torch.stack(outputs.scores, 1)

        return completions, logits