import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler

from dataset import MATH
from model import PolicyModel
from trainer import SCoRETrainer

from transformers import AutoTokenizer

from config import config  # Import configurations

def train_model():
    tokenizer = AutoTokenizer.from_pretrained(config['policy_model_name'])
    policy_model = PolicyModel()
    ref_model = PolicyModel()
    train_dataset = MATH(tokenizer)
    trainer = SCoRETrainer(config, policy_model, ref_model, train_dataset)
    trainer.train()