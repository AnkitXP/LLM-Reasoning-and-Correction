import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler

from dataset import MATH
from model import PolicyModel
# from trainer import Trainer

from config import config  # Import configurations

def train():
    # Initialize the model
    policy_model = PolicyModel(trainable=False)

    # Load dataset and DataLoader
    train_dataset = MATH(data_dir='data/MATH', split='train')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    query = "[INST]Good Morning! How are you doing?[\INST]"
    print(policy_model.generate(query))

    # # Prepare optimizer and scheduler
    # optimizer = AdamW(model.parameters(), lr=config['lr'])
    # num_training_steps = len(train_loader) * config['epochs']
    # scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # # Training loop
    # for epoch in range(config['epochs']):
    #     print(f"Epoch {epoch + 1}/{config['epochs']}")
    #     epoch_loss = 0.0

    #     # Progress bar
    #     progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
    #     for step, batch in progress_bar:
    #         optimizer.zero_grad()

    #         # Prepare inputs and labels
    #         inputs = tokenizer(batch['problem'], return_tensors='pt', padding=True, truncation=True, max_length=config['seq_length']).input_ids.to(device)
    #         labels = tokenizer(batch['solution'], return_tensors='pt', padding=True, truncation=True, max_length=config['seq_length']).input_ids.to(device)

    #         # Forward pass
    #         outputs = model(input_ids=inputs, labels=labels)
    #         loss = outputs.loss

    #         # Backward pass and optimization
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()

    #         # Update progress bar
    #         epoch_loss += loss.item()
    #         progress_bar.set_postfix({'loss': loss.item()})

    #     avg_epoch_loss = epoch_loss / len(train_loader)
    #     print(f"Average Loss for Epoch {epoch + 1}: {avg_epoch_loss:.4f}")

    #     # Save model at intervals
    #     if (epoch + 1) % config['save_interval'] == 0:
    #         save_path = os.path.join(config['save_dir'], f'mistral-epoch-{epoch + 1}')
    #         os.makedirs(save_path, exist_ok=True)
    #         policy_model.save_model(save_path, model_name=f"mistral_epoch_{epoch + 1}")

    # # Final save
    # policy_model.save_model(config['save_dir'], model_name="final_mistral_model")