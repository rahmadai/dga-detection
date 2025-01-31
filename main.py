import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from torch import nn
from sklearn.metrics import confusion_matrix
import numpy as np
from transformers import BertModel
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.models import StaticEmbedding

from data.domaindataset import DomainDataset
from data.dataloader import DataLoader
from preprocessing.phonetic_preprocessor import PhoneticPreprocessor
from preprocessing.semantic_preprocessor import SemanticPreprocessor
from models.dga_detector import DGA_Detection_Model
import pickle
from loguru import logger

# Set random seed for reproducibility
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load a model from the HuggingFace hub
bge_static_embedding = StaticEmbedding.from_model2vec("FlukeTJ/bge-m3-m2v-distilled-256")
# Define Hyperparameters
semantic_model = SentenceTransformer(modules=[bge_static_embedding], device=device)
phonetic_vocab_size = 512
embedding_dim = 128

# 0 for save dataloader
# 1 for load dataloader & train model
train_model = 1

if train_model == 1:
    logger.info("Starting data preprocessing...")
    # Instantiate DataLoader (your custom DataLoader, not torch.utils.data.DataLoader)
    data_loader = DataLoader(benign_path='data/benign_10k.csv', dga_path='data/dga_10k.csv')
    data_loader.load_and_split_data()
    train_df = data_loader.get_train_df()
    val_df = data_loader.get_val_df()
    test_df = data_loader.get_test_df() # Get test data
    # train_df = train_df.head(10000)
    # val_df = val_df.head(1000)
    # test_df = test_df.head(1000) # Limit test data for faster testing

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
    
    test_phonetic_tokens = []
    test_phonetic_masks = []
    for domain in test_df['domain']:
        tokens, masks = phonetic_preprocessor.tokenize_domain(domain)
        test_phonetic_tokens.append(tokens)
        test_phonetic_masks.append(masks)

    # Extract and scale phonetic features
    logger.info("Start extract phonetic features...")
    train_phonetic_features = phonetic_preprocessor.extract_features(train_df)
    val_phonetic_features = phonetic_preprocessor.extract_features(val_df)
    test_phonetic_features = phonetic_preprocessor.extract_features(test_df)
    train_phonetic_scaled, val_phonetic_scaled, test_phonetic_scaled = phonetic_preprocessor.scale_features(train_phonetic_features, val_phonetic_features, test_phonetic_features)

    # Instantiate SemanticPreprocessor
    semantic_preprocessor = SemanticPreprocessor()

    # Tokenize domains
    train_semantic_embeds = []
    logger.info("Start tokenize embedding domains...")
    for domain in train_df['domain']:
        embed = semantic_preprocessor.tokenize_domain(domain)
        train_semantic_embeds.append(embed)

    val_semantic_embeds = []
    for domain in val_df['domain']:
        embed = semantic_preprocessor.tokenize_domain(domain)
        val_semantic_embeds.append(embed)

    test_semantic_embeds = []
    for domain in test_df['domain']:
        embed = semantic_preprocessor.tokenize_domain(domain)
        test_semantic_embeds.append(embed)

    # Load dataset
    train_dataset = DomainDataset(train_df, train_phonetic_tokens, train_phonetic_masks, train_semantic_embeds, train_phonetic_scaled)
    val_dataset = DomainDataset(val_df, val_phonetic_tokens, val_phonetic_masks, val_semantic_embeds, val_phonetic_scaled)
    test_dataset = DomainDataset(test_df, test_phonetic_tokens, test_phonetic_masks, test_semantic_embeds, test_phonetic_scaled) # Create test dataset

    logger.info("Data preprocessing completed.")

    logger.info("Loading datasets and starting model training...")

    # Instantiate model
    model = DGA_Detection_Model(phonetic_vocab_size, embedding_dim, semantic_model)

    # Set up training parameters
    
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create DataLoaders
    train_dataloader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = TorchDataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = TorchDataLoader(test_dataset, batch_size=32, shuffle=False) # Create test dataloader
    
    # Train and validate the model
    logger.info("Starting training...")
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dataloader: # Use train_dataloader
            phonetic_token_ids = batch['phonetic_token_ids'].to(device)
            semantic_embeds = batch['semantic_token_ids'].to(device) # Move to device
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(phonetic_token_ids, semantic_embeds)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * phonetic_token_ids.size(0)
        
        train_loss = train_loss / len(train_dataloader.dataset) # use train_dataloader
        logger.info(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader: # Use val_dataloader
                phonetic_token_ids = batch['phonetic_token_ids'].to(device)
                semantic_embeds = batch['semantic_token_ids'].to(device)  # Move to device
                labels = batch['labels'].to(device)
                
                outputs = model(phonetic_token_ids, semantic_embeds)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item() * phonetic_token_ids.size(0)
        
        val_loss = val_loss / len(val_dataloader.dataset) # use val_dataloader
        logger.info(f'Epoch: {epoch+1}, Validation Loss: {val_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'dga_detection_model.pth')
    logger.info("Model training completed and model saved.")

    # --- Start of Inference/Testing Code ---
    logger.info("Starting model inference/testing...")

    model.eval()  # Set model to evaluation mode
    all_predicted_labels = []
    all_true_labels = []

    with torch.no_grad():  # Disable gradient calculations during inference
        for batch in test_dataloader:
            phonetic_token_ids = batch['phonetic_token_ids'].to(device)
            semantic_embeds = batch['semantic_token_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(phonetic_token_ids, semantic_embeds)
            predicted_labels = (outputs.squeeze() > 0.5).float() # Convert probabilities to binary predictions
            
            all_predicted_labels.extend(predicted_labels.cpu().numpy())  # Move to CPU and convert to numpy
            all_true_labels.extend(labels.cpu().numpy())

    # Generate Confusion Matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1_score}")
    logger.info("Inference completed.")