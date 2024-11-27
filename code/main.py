import argparse
import torch
import numpy as np
import random
import os

from train import train_model

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
    parser.add_argument('--seed', type=int, default=44,
                        help="seed", required=False)
    return parser.parse_args()
    
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    args = parse_arguments()

    set_seed(args.seed)

    if args.task == 'train':
        train_model()        
    else:
        raise ValueError('Invalid Task')