import torch
import numpy as np
from data.dataloader import DGADatasetLoader
from preprocessing.semantic_preprocessor import SemanticPreprocessor
from preprocessing.phonetic_preprocessor import PhoneticPreprocessor
from models.dga_detector import DGADetector
from utils.metrics import evaluate_model
from torch import nn

def main():
    # Load and preprocess data
    dataset_loader = DGADatasetLoader()
    X_train, X_test, y_train, y_test = dataset_loader.load_datasets()
    
    # Extract semantic features
    semantic_preprocessor = SemanticPreprocessor()
    semantic_features_train = semantic_preprocessor.extract_semantic_features(X_train)
    semantic_features_test = semantic_preprocessor.extract_semantic_features(X_test)
    
    # Extract phonetic features
    phonetic_preprocessor = PhoneticPreprocessor()
    phonetic_features_train = phonetic_preprocessor.extract_phonetic_features(X_train)
    phonetic_features_test = phonetic_preprocessor.extract_phonetic_features(X_test)
    
    # Combine features
    X_train_combined = np.hstack([semantic_features_train, phonetic_features_train])
    X_test_combined = np.hstack([semantic_features_test, phonetic_features_test])
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_combined)
    X_test_tensor = torch.FloatTensor(X_test_combined)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Initialize model
    model = DGADetector(input_dim=X_train_combined.shape[1])
    
    # Training setup (simplified)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop (basic implementation)
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    with torch.no_grad():
        test_predictions = model(X_test_tensor).numpy()
    
    # Compute metrics
    metrics = evaluate_model(y_test, test_predictions)
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()