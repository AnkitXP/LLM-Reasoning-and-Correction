import os
import torch
from dataset import MATH
from model import PolicyModel
from trainer import SCoRETrainer

from config import config  # Import configurations

def train_model():
    """
    Trains a policy model using the MATH dataset and the SCoRE reinforcement learning approach.

    """

    print("<===================================== Training ====================================>")
    #Instantiate Policy and Reference Models
    policy_model = PolicyModel()
    ref_model = PolicyModel().eval()

    #Create Dataset
    train_dataset = MATH()

    #Instantiate trainer and initiate training
    trainer = SCoRETrainer(config, policy_model, ref_model, train_dataset)
    trainer.train()

def evaluate_model(model_name: str, tokenizer_name: str):
    """
    Evaluates the trained model on the test dataset.

    Args:
        model_name (str): The name of the model directory from which to load the saved model.
        tokenizer_name (str): The name of the tokenizer directory from which to load the tokenizer.
    """

    print("<===================================== Testing ====================================>")
    
    # Load the test dataset
    print("Loading the test dataset...")
    test_dataset = MATH(split='test')  # Assuming MATH dataset follows Hugging Face's Dataset structure.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load the saved policy model
    print(f"Loading model from {model_name}...")
    policy_model = PolicyModel.from_pretrained(model_name)
    policy_model.to(device)
    policy_model.eval()

    # Create DataLoader for the test dataset
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    predictions, references = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            # Prepare inputs
            inputs = tokenizer(batch['problem'], padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Generate model outputs
            outputs = policy_model.generate(**inputs)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_outputs)

            # Store references for evaluation
            references.extend(batch['solution'])  # Assuming test_dataset includes 'solution'

    # Evaluate predictions using an equivalence check or accuracy metric
    print("Calculating evaluation metrics...")
    correct = sum(1 for pred, ref in zip(predictions, references) if is_equiv(pred, ref))
    total = len(references)
    accuracy = correct / total * 100

    # Log the results
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

    print(f"Evaluation Metrics:\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    
    