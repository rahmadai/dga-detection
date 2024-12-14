import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, benign_path, dga_path):
        self.benign_path = benign_path
        self.dga_path = dga_path
        self.train_df = None
        self.val_df = None
    
    def load_and_split_data(self, test_size=0.2, random_state=42):
        # Load datasets
        benign_df = pd.read_csv(self.benign_path)
        dga_df = pd.read_csv(self.dga_path)
        
        # Add labels
        benign_df['label'] = 0
        dga_df['label'] = 1
        
        # Combine datasets
        full_df = pd.concat([benign_df, dga_df], ignore_index=True)
        
        # Split into train and validation sets
        self.train_df, self.val_df = train_test_split(full_df, test_size=test_size, random_state=random_state)
    
    def get_train_df(self):
        return self.train_df
    
    def get_val_df(self):
        return self.val_df