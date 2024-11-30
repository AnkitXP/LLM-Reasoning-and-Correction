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
    
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    args = parse_arguments()

    set_seed(config['seed'])

    if args.task == 'train':
        download_and_save_model(config['policy_model_name'], config['model_dir'])
        train_model()
    elif args.task == 'evaluate':
        evaluate_model()        
    else:
        raise ValueError('Invalid Task')