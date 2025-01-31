import pandas as pd
from sklearn.model_selection import train_test_split
import wordninja

class DataLoader:
    def __init__(self, benign_path, dga_path):
        self.benign_path = benign_path
        self.dga_path = dga_path
        self.train_df = None
        self.val_df = None
        self.test_df = None
    
    def load_and_split_data(self, train_size=0.8, val_test_size=0.2, val_size_ratio=0.5, random_state=42):
        # Load datasets
        benign_df = pd.read_csv(self.benign_path)
        dga_df = pd.read_csv(self.dga_path)

        # Extract domain name without TLD
        benign_df['domain'] = benign_df['domain'].apply(lambda x: x.split('.')[0])
        dga_df['domain'] = dga_df['domain'].apply(lambda x: x.split('.')[0])

        # Split by word using wordninja
        benign_df['domain'] = benign_df['domain'].apply(lambda x: ' '.join(wordninja.split(x)))
        dga_df['domain'] = dga_df['domain'].apply(lambda x: ' '.join(wordninja.split(x)))

        # Add labels
        benign_df['label'] = 0
        dga_df['label'] = 1
        
        # Combine datasets
        full_df = pd.concat([benign_df, dga_df], ignore_index=True)
        
        # Ensure sizes are valid
        if train_size + val_test_size != 1.0:
           raise ValueError("Train and val+test sizes must sum to 1.")
        if val_size_ratio < 0 or val_size_ratio > 1:
            raise ValueError("Val size ratio must be between 0 and 1")

        # Split into train and combined validation/test sets
        self.train_df, val_test_df = train_test_split(full_df, train_size=train_size, random_state=random_state)
        
        # Split the validation/test set into validation and test sets
        self.val_df, self.test_df = train_test_split(val_test_df, test_size=1-val_size_ratio, random_state=random_state)
    
    def get_train_df(self):
        return self.train_df
    
    def get_val_df(self):
        return self.val_df

    def get_test_df(self):
        return self.test_df