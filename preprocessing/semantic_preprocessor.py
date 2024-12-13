import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class SemanticPreprocessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
    
    def segment_domain(self, domain):
        # Split domain into meaningful components
        return domain.replace('.', ' ').split()
    
    def extract_semantic_features(self, domains):
        features = []
        for domain in domains:
            # Segment domain
            segments = self.segment_domain(domain)
            
            # Tokenize and get embeddings
            inputs = self.tokenizer(segments, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Aggregate embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Compute mean and std
            feature_vector = torch.cat([
                embeddings.mean(dim=0), 
                embeddings.std(dim=0)
            ])
            
            features.append(feature_vector.numpy())
        
        return np.array(features)