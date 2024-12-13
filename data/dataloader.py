import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DGADatasetLoader:
    def __init__(self, dga_path='data/dga_domains.csv', benign_path='data/benign_domains.csv'):
        self.dga_path = dga_path
        self.benign_path = benign_path
    
    def load_datasets(self, test_size=0.2, random_state=42):
        # Load DGA domains
        dga_df = pd.read_csv(self.dga_path)
        dga_df['label'] = 1  # DGA domains labeled as 1
        
        # Load benign domains
        benign_df = pd.read_csv(self.benign_path)
        benign_df['label'] = 0  # Benign domains labeled as 0
        
        # Combine datasets
        combined_df = pd.concat([dga_df, benign_df], ignore_index=True)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            combined_df['domain'], 
            combined_df['label'], 
            test_size=test_size, 
            random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test