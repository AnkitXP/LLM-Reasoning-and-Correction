import json
import os
from torch.utils.data import Dataset

class MATH(Dataset):
    def __init__(self, data_dir = '../data/MATH', split='train', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.data = self.load_data()
    
    def load_data(self):
        
        data = []
        for root, _, files in os.walk(self.data_dir):
            for file_name in files:
                if file_name.endswith('.json'):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        data.append(content)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        item = self.data[idx]
        
        # Extract relevant information from the JSON data
        problem = item['problem']
        level = item['level']
        problem_type = item['type']
        solution = item['solution']
        
        # Create a sample dictionary
        sample = {
            'problem': problem,
            'level': level,
            'type': problem_type,
            'solution': solution
        }
        
        # Apply any transformations if specified
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

if __name__ == '__main__':
    math = MATH(split='train')
    print(math[0])