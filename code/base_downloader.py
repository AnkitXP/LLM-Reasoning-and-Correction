import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_and_save_model(model_name: str, save_dir: str):
    """
    Downloads a Hugging Face model and tokenizer and saves them to a specified directory.

    Args:
        model_name (str): The name of the Hugging Face model to download.
        save_dir (str): The directory where the model and tokenizer will be saved.
    """
    # Ensure the save directory exists
    parent_dir = os.path.join(os.getcwd(), os.pardir)
    save_dir = os.path.join(parent_dir, save_dir)
    save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Download and save the model
    print(f"Downloading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    # Download and save the tokenizer
    print(f"Downloading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to {save_dir}")