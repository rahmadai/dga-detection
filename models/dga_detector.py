import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DGADetector(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(DGADetector, self).__init__()
        
        # 1D Convolutional Layer
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        
        # Self-Attention Mechanism
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # 1D Convolution
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1d(x))
        x = x.squeeze(1)
        
        # Self-Attention
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention Scores
        attention_scores = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.size(-1)), dim=-1)
        
        # Weighted sum
        attended_values = torch.matmul(attention_scores, V)
        
        # Classification
        x = F.relu(self.fc1(attended_values.mean(dim=1)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x