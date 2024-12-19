import pandas as pd
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from data.domaindataset import DomainDataset
import torch
from transformers import BertModel
from data.dataloader import DataLoader
from preprocessing.phonetic_preprocessor import PhoneticPreprocessor
from preprocessing.semantic_preprocessor import SemanticPreprocessor
from models.dga_detector import DGA_Detection_Model
from torch import nn
import pickle
from loguru import logger

# Set random seed for reproducibility
torch.manual_seed(42)

# Define Hyperparameters
semantic_model = BertModel.from_pretrained('bert-base-uncased')
phonetic_vocab_size = 512
embedding_dim = 128

# 0 for save dataloader
# 1 for load dataloader & train model
train_model = 0

if train_model == 0:
    logger.info("Starting data preprocessing...")
    # Instantiate DataLoader
    data_loader = DataLoader(benign_path='data/benign_10k.csv', dga_path='data/dga_10k.csv')
    data_loader.load_and_split_data()
    train_df = data_loader.get_train_df()
    val_df = data_loader.get_val_df()

    # Instantiate PhoneticPreprocessor
    logger.info("Training phonetic tokenizer...")
    save_phonetic_model = 'results/bpe_models'
    phonetic_preprocessor = PhoneticPreprocessor(vocab_size=512)
    train_domains = train_df['domain'].tolist()
    phonetic_preprocessor.train_tokenizer(train_domains, save_phonetic_model)
    logger.info("Phonetic tokenizer trained and saved.")

    # Tokenize domains
    train_phonetic_tokens = []
    train_phonetic_masks = []
    logger.info("Start tokenize phonetic tokens...")
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
    logger.info("Start extract phonetic features...")
    train_phonetic_features = phonetic_preprocessor.extract_features(train_df)
    val_phonetic_features = phonetic_preprocessor.extract_features(val_df)
    train_phonetic_scaled, val_phonetic_scaled = phonetic_preprocessor.scale_features(train_phonetic_features, val_phonetic_features)

    # Instantiate SemanticPreprocessor
    semantic_preprocessor = SemanticPreprocessor()

    # Tokenize domains
    train_semantic_tokens = []
    train_semantic_masks = []
    logger.info("Start tokenize embedding domains...")
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

    # Save train dataset into pickle
    logger.info("Saving train dataset into pickle...")
    with open('results/train_dataset.pkl', 'wb') as f:
        train_dataset = DomainDataset(train_df, train_phonetic_tokens, train_phonetic_masks, train_semantic_tokens, train_semantic_masks, train_phonetic_scaled)
        pickle.dump(train_dataset, f)

    # Save validation dataset into pickle
    logger.info("Saving validation dataset into pickle...")
    with open('results/val_dataset.pkl', 'wb') as f:
        val_dataset = DomainDataset(val_df, val_phonetic_tokens, val_phonetic_masks, val_semantic_tokens, val_semantic_masks, val_phonetic_scaled)
        pickle.dump(val_dataset, f)

    logger.info("Data preprocessing completed.")

elif train_model == 1:
    logger.info("Loading datasets and starting model training...")
    # Load train dataset from pickle
    with open('results/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)

    # Load validation dataset from pickle
    with open('results/val_dataset.pkl', 'rb') as f:
        val_dataset = pickle.load(f)

    logger.info("Loading data loaders...")
    train_loader = TorchDataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=16, shuffle=False)

    # Instantiate model
    model = DGA_Detection_Model(phonetic_vocab_size, embedding_dim, semantic_model)

    # Set up training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train and validate the model
    logger.info("Starting training...")
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
        logger.info(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')
        
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
        logger.info(f'Epoch: {epoch+1}, Validation Loss: {val_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'dga_detection_model.pth')
    logger.info("Model training completed and model saved.")