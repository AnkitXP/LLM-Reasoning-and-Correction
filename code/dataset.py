import json
import os
from torch.utils.data import Dataset

class MATH(Dataset):
    """
    Creates the Math Dataset from the JSON files and the base for the dataloader
    """

    def __init__(self, data_dir = 'data/MATH', split='train'):
        self.data_dir = os.path.join(data_dir, split)
        self.data = self.load_data()
    
    def load_data(self):
        
        data = []
        for root, _, files in os.walk(self.data_dir):
            for file_name in files:
                if file_name.endswith('.json'):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        content.pop('level')
                        content.pop('type')
                        data.append(content)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        item = self.data[idx]
                
        return item['problem'], item['solution']