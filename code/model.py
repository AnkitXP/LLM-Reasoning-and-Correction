import torch
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import config
from utils import pad

class PolicyModel():
    def __init__(self):
        """
        Initialize the policy model and tokenizer
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        parent_dir = os.path.join(os.getcwd(), os.pardir)
        model_dir = os.path.join(parent_dir, config['model_dir'])
        model_path = os.path.join(model_dir, config['policy_model_name'])
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, input_ids, attention_mask=None, **gen_kwargs):
        """
        Generates only the completion and respective logits based on input length
        
        Returns tensors padded responses [batch size, sequence length] and padded logits [batch size, sequence length, vocabulary size]
        """

        responses = []
        logitss = []
        batch_size = input_ids.shape[0]

        for i in range(0, batch_size, config['local_rollout_forward_batch_size']):
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
            
            responses.append(completions)
            logitss.append(logits)

        # padding tensors
        padded_responses = pad(responses, padding_value=self.tokenizer.pad_token_id, padding_side="right")
        padded_logitss = pad(logitss, padding_value=0, padding_side="right")

        # reshaping
        padded_responses = padded_responses.view(-1, padded_responses.shape[-1])[:batch_size]
        padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]
        
        return padded_responses, padded_logitss
    
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
        
        prompts_tokenized = self.tokenizer.apply_chat_template(
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
        
        prompts_tokenized = self.tokenizer.apply_chat_template(
            conversation=second_messages,
            tools=None,
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return second_messages, prompts_tokenized

    def save_model(self):
        """
        Saves the model based on the save intervals.
        """
        model_dir = config['save_dir']+'/SCoRE-' + config['policy_model_name']
        parent_dir = os.path.join(os.getcwd(), os.pardir)
        save_dir = os.path.join(parent_dir, model_dir)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f'Model and Tokenizer saved to {save_dir}')