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