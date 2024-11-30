import os
import torch
from dataset import MATH
from model import PolicyModel
from trainer import SCoRETrainer
import gc

from utils import check_correct
from config import config  # Import configurations

def train_model():
    """
    Trains a policy model using the MATH dataset and the SCoRE reinforcement learning approach.

    """

    print("<===================================== Training ====================================>")
    #Instantiate Policy and Reference Models
    policy_model = PolicyModel()
    ref_model = PolicyModel()

    #Create Dataset
    train_dataset = MATH()

    #Instantiate trainer and initiate training
    trainer = SCoRETrainer(config, policy_model, ref_model, train_dataset)
    trainer.train()

def evaluate_model():
    """
    Evaluates the trained model on the test dataset.
    """
    print("<===================================== Testing ====================================>")
    
    # Load the test dataset
    print("Loading the test dataset...")
    test_dataset = MATH(split='test')  # Assuming MATH dataset follows Hugging Face's Dataset structure.

    # Load the saved policy model
    print(f"Loading model from {config['policy_model_name']}...")
    policy_model = PolicyModel()

    # Create DataLoader for the test dataset
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    t1_correct = []
    t2_correct = []

    with torch.no_grad():
        for problems_batch, solutions_batch in test_dataloader:
            # Prepare inputs
            # First attempt template
            first_messages, tokenized_first_prompts = policy_model.prepare_first_attempt_input(
                                                                                config['first_attempt_prompt'], 
                                                                                problems_batch
                                                                                )

            # First attempt policy completions
            first_outputs, _ = policy_model.generate(
                                                    tokenized_first_prompts['input_ids'].to(policy_model.device), 
                                                    tokenized_first_prompts['attention_mask'].to(policy_model.device),
                                                    **config['gen_kwargs']
                                                    )
                        
            
            # decode first attempt outputs
            first_decoded_completions = policy_model.tokenizer.batch_decode(first_outputs, skip_special_tokens=True) 

            # check first attempt correctness
            t1_correct.extend(check_correct(first_decoded_completions, solutions_batch))

            # Cleanup first attempt variables
            del _, first_outputs
            torch.cuda.empty_cache()
            
            # Second attempt template
            _, tokenized_second_prompts = policy_model.prepare_second_attempt_input(
                                                                        first_messages, 
                                                                        first_decoded_completions, 
                                                                        config['second_attempt_prompt']
                                                                        )
            # second attempt policy completions
            second_outputs, second_logits = policy_model.generate(
                                        tokenized_second_prompts['input_ids'].to(policy_model.device),
                                        tokenized_second_prompts['attention_mask'].to(policy_model.device),
                                        **config['gen_kwargs']
                                        )
    

            del _, tokenized_second_prompts, first_messages, first_decoded_completions

            # decode second attempt outputs
            second_decoded_completions = policy_model.tokenizer.batch_decode(second_outputs, skip_special_tokens=True)

            # check second attempt correctness
            t2_correct.extend(check_correct(second_decoded_completions, solutions_batch))

            # Cleanup second attempt variables
            del _, second_logits, second_outputs, second_decoded_completions
            gc.collect()

    # Evaluate predictions using an equivalence check or accuracy metric
    print("Calculating evaluation metrics...")

    total = len(t1_correct)

    # Attempt 1 and 2 accuracy
    t1_acc = sum(t1_correct) / total * 100    
    t2_acc = sum(t2_correct) / total * 100

    delta = t2_acc - t1_acc

    # TODO:Log the results?
    metrics = {
        "t1_accuracy": t1_acc,
        "t2_accuracy": t2_acc,
        "t1_correct" : t1_correct,
        "t2_correct" : t2_correct,
        "t1_t2_delta": delta,
        "total"      : total
    }

    print(f"Evaluation Metrics:\nFirst attempt accuracy: {t1_acc:.2f}% ({t1_correct}/{total})")
    print(f"Final Accuracy: {t2_acc:.2f}% ({t2_correct}/{total})")
    
    