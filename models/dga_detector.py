import torch.nn as nn
import torch
from transformers import BertModel

class DGA_Detection_Model(nn.Module):
    def __init__(self, phonetic_vocab_size, embedding_dim, semantic_model):
        super(DGA_Detection_Model, self).__init__()
        self.phonetic_embedding = nn.Embedding(phonetic_vocab_size, embedding_dim)
        self.semantic_model = semantic_model
        self.fc_phonetic = nn.Linear(embedding_dim, 128)
        self.fc_semantic = nn.Linear(768, 128)
        self.fc_combined = nn.Linear(256, 64)
        self.fc_output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, phonetic_token_ids, semantic_token_ids, semantic_attention_mask):
        # Phonetic embedding
        phonetic_embed = self.phonetic_embedding(phonetic_token_ids)
        phonetic_embed = phonetic_embed.mean(dim=1)  # Mean pooling
        phonetic_embed = self.fc_phonetic(phonetic_embed)
        
        # Semantic embedding
        semantic_outputs = self.semantic_model(input_ids=semantic_token_ids, attention_mask=semantic_attention_mask)
        semantic_embed = semantic_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        semantic_embed = self.fc_semantic(semantic_embed)
        
        # Concatenate embeddings
        concatenated = torch.cat((phonetic_embed, semantic_embed), dim=1)
        
        # Classifier layers
        x = nn.functional.relu(self.fc_combined(concatenated))
        x = self.fc_output(x)
        x = self.sigmoid(x)
        return x