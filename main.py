import pandas as pd
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
import torch
from transformers import BertModel
from data.dataloader import DataLoader
from preprocessing.phonetic_preprocessor import PhoneticPreprocessor
from preprocessing.semantic_preprocessor import SemanticPreprocessor
from models.dga_detector import DGA_Detection_Model
from torch import nn

# Instantiate DataLoader
data_loader = DataLoader(benign_path='data/benign_domains.csv', dga_path='data/dga_domains.csv')
data_loader.load_and_split_data()
train_df = data_loader.get_train_df()
val_df = data_loader.get_val_df()

# Instantiate PhoneticPreprocessor
phonetic_preprocessor = PhoneticPreprocessor(vocab_size=500)
train_domains = train_df['domain'].tolist()
phonetic_preprocessor.train_tokenizer(train_domains)

# Tokenize domains
train_phonetic_tokens = []
train_phonetic_masks = []
for domain in train_df['domain']:
    tokens, masks = phonetic_preprocessor.tokenize_domain(domain)
    train_phonetic_tokens.append(tokens)
    train_phonetic_masks.append(masks)

val_phonetic_tokens = []
val_phonetic_masks = []
for domain in val_df['domain']:
    tokens, masks = phonetic_preprocessor.tokenize_domain(domain)
    val_phonetic_tokens.append(tokens)
    val_phonetic_masks.append(masks)

# Extract and scale phonetic features
train_phonetic_features = phonetic_preprocessor.extract_features(train_df)
val_phonetic_features = phonetic_preprocessor.extract_features(val_df)
train_phonetic_scaled, val_phonetic_scaled = phonetic_preprocessor.scale_features(train_phonetic_features, val_phonetic_features)

# Instantiate SemanticPreprocessor
semantic_preprocessor = SemanticPreprocessor()

# Tokenize domains
train_semantic_tokens = []
train_semantic_masks = []
for domain in train_df['domain']:
    tokens, masks = semantic_preprocessor.tokenize_domain(domain)
    train_semantic_tokens.append(tokens)
    train_semantic_masks.append(masks)

val_semantic_tokens = []
val_semantic_masks = []
for domain in val_df['domain']:
    tokens, masks = semantic_preprocessor.tokenize_domain(domain)
    val_semantic_tokens.append(tokens)
    val_semantic_masks.append(masks)

# Create custom Dataset
class DomainDataset(Dataset):
    def __init__(self, df, phonetic_tokens, phonetic_masks, semantic_tokens, semantic_masks, phonetic_features):
        self.df = df
        self.phonetic_tokens = phonetic_tokens
        self.phonetic_masks = phonetic_masks
        self.semantic_tokens = semantic_tokens
        self.semantic_masks = semantic_masks
        self.phonetic_features = phonetic_features
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        phonetic_token_ids = torch.tensor(self.phonetic_tokens[idx], dtype=torch.long)
        phonetic_attention_mask = torch.tensor(self.phonetic_masks[idx], dtype=torch.long)
        semantic_token_ids = torch.tensor(self.semantic_tokens[idx], dtype=torch.long)
        semantic_attention_mask = torch.tensor(self.semantic_masks[idx], dtype=torch.long)
        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float)
        phonetic = torch.tensor(self.phonetic_features[idx], dtype=torch.float)
        return {
            'phonetic_token_ids': phonetic_token_ids,
            'phonetic_attention_mask': phonetic_attention_mask,
            'semantic_token_ids': semantic_token_ids,
            'semantic_attention_mask': semantic_attention_mask,
            'labels': label,
            'phonetic_features': phonetic
        }

train_dataset = DomainDataset(train_df, train_phonetic_tokens, train_phonetic_masks, train_semantic_tokens, train_semantic_masks, train_phonetic_scaled)
val_dataset = DomainDataset(val_df, val_phonetic_tokens, val_phonetic_masks, val_semantic_tokens, val_semantic_masks, val_phonetic_scaled)

train_loader = TorchDataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = TorchDataLoader(val_dataset, batch_size=16, shuffle=False)

# Instantiate model
semantic_model = BertModel.from_pretrained('bert-base-uncased')
phonetic_vocab_size = 500
embedding_dim = 128
model = DGA_Detection_Model(phonetic_vocab_size, embedding_dim, semantic_model)

# Set up training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train and validate the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        phonetic_token_ids = batch['phonetic_token_ids'].to(device)
        semantic_token_ids = batch['semantic_token_ids'].to(device)
        semantic_attention_mask = batch['semantic_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(phonetic_token_ids, semantic_token_ids, semantic_attention_mask)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * phonetic_token_ids.size(0)
    
    train_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            phonetic_token_ids = batch['phonetic_token_ids'].to(device)
            semantic_token_ids = batch['semantic_token_ids'].to(device)
            semantic_attention_mask = batch['semantic_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(phonetic_token_ids, semantic_token_ids, semantic_attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            
            val_loss += loss.item() * phonetic_token_ids.size(0)
    
    val_loss = val_loss / len(val_loader.dataset)
    print(f'Epoch: {epoch+1}, Validation Loss: {val_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'dga_detection_model.pth')