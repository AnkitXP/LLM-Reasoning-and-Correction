import argparse
import torch
import numpy as np
import random
import os

from model_runner import train_model, evaluate_model
from base_downloader import download_and_save_model
from config import config

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,
                        help='name of this task: train/generate', required=True)
    return parser.parse_args()

def print_selected_config(config):
    keys_to_print = [
        'policy_model_name', 'seed', 'batch_size', 'local_rollout_forward_batch_size',
        'gradient_accumulation_steps', 'total_episodes', 'alpha', 'beta_one',
        'beta_two', 'lr', 'gen_kwargs'
    ]

    for key in keys_to_print:
        if key in config:
            if isinstance(config[key], dict):  # If the value is a dictionary, print it recursively
                print(f"{key}:")
                for sub_key, sub_value in config[key].items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {config[key]}")
    
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    args = parse_arguments()

    set_seed(config['seed'])
    print_selected_config(config)

    if args.task == 'download':
        download_and_save_model(
                                config['policy_model_name'], 
                                config['model_dir']
                                )

    elif args.task == 'train':
        train_model()

    elif args.task == 'evaluate':
        evaluate_model()        
    
    else:
        raise ValueError('Invalid Task')