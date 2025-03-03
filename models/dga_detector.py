import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

class DGA_Detection_Model(nn.Module):
    def __init__(self, phonetic_vocab_size, embedding_dim, semantic_model):
        super(DGA_Detection_Model, self).__init__()
        self.phonetic_embedding = nn.Embedding(phonetic_vocab_size, embedding_dim)
        self.semantic_model = semantic_model
        self.fc_phonetic = nn.Linear(embedding_dim, 128)
        self.fc_semantic = nn.Linear(256, 128)
        self.fc_combined = nn.Linear(128 + 128, 64)  # Adjusted input size
        self.fc_output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    # def __init__(self, phonetic_embedding_dim, semantic_embedding_dim, semantic_model):
    #     super(DGA_Detection_Model, self).__init__()
    #     self.phonetic_embedding = nn.Embedding(phonetic_embedding_dim, phonetic_embedding_dim/2)
    #     self.semantic_model = semantic_model
    #     self.fc_phonetic = nn.Linear(phonetic_embedding_dim, phonetic_embedding_dim/2)
    #     self.fc_semantic = nn.Linear(semantic_embedding_dim, semantic_embedding_dim/2)
    #     self.fc_combined = nn.Linear(phonetic_embedding_dim + semantic_embedding_dim, 64)  # Adjusted input size
    #     self.fc_output = nn.Linear(64, 1)
    #     self.sigmoid = nn.Sigmoid()
    
    def forward(self, phonetic_token, semantic_embed):
        # Phonetic embedding
        phonetic_embed_raw = self.phonetic_embedding(phonetic_token)
        phonetic_embed = phonetic_embed_raw.mean(dim=1)  # Mean pooling
        phonetic_embed = self.fc_phonetic(phonetic_embed)
        
        # Semantic embedding (already processed)
        semantic_embed = self.fc_semantic(semantic_embed)
        
        # Concatenate embeddings
        concatenated = torch.cat((phonetic_embed, semantic_embed), dim=1)
        
        # Classifier layers
        x = nn.functional.relu(self.fc_combined(concatenated))
        x = self.fc_output(x)
        x = self.sigmoid(x)
        return x