from torch.utils.data import DataLoader as TorchDataLoader, Dataset
import torch

# Create custom Dataset
class DomainDataset(Dataset):
    def __init__(self, df, phonetic_tokens, phonetic_masks, semantic_embeddings, phonetic_features):
        self.df = df
        self.phonetic_tokens = phonetic_tokens
        self.phonetic_masks = phonetic_masks
        self.semantic_embeddings = semantic_embeddings
        self.phonetic_features = phonetic_features
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        phonetic_token_ids = torch.tensor(self.phonetic_tokens[idx], dtype=torch.long)
        phonetic_attention_mask = torch.tensor(self.phonetic_masks[idx], dtype=torch.long)
        semantic_embedding = torch.tensor(self.semantic_embeddings[idx], dtype=torch.float)
        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float)
        phonetic = torch.tensor(self.phonetic_features[idx], dtype=torch.float)
        return {
            'phonetic_token_ids': phonetic_token_ids,
            'phonetic_attention_mask': phonetic_attention_mask,
            'semantic_token_ids': semantic_embedding,
            'labels': label,
            'phonetic_features': phonetic
        }