import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.models import StaticEmbedding

from data.domaindataset import DomainDataset  # Assuming you have this
from preprocessing.phonetic_preprocessor import PhoneticPreprocessor  # Assuming you have this
from preprocessing.semantic_preprocessor import SemanticPreprocessor
from models.dga_detector import DGA_Detection_Model # Assuming you have this
import pickle
from loguru import logger
import numpy as np
import wordninja  # Import wordninja
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Set random seed for reproducibility
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_config = "P512_S256" # or whatever config your model was trained on

# Configure logger
logger.add(f"logs/inference_{model_config}.log", level="INFO")  # For detailed inference logs
logger.add(f"logs/errors_inference_{model_config}.log", level="ERROR") # For errors
logger.info(f"Using device: {device}")


# Load a model from the HuggingFace hub
bge_static_embedding = StaticEmbedding.from_model2vec("FlukeTJ/bge-m3-m2v-distilled-256")
# Define Hyperparameters
semantic_model = SentenceTransformer(modules=[bge_static_embedding], device=device)
phonetic_embedding_size = 512
semantic_embedding_size = 128

class InferenceDataset(Dataset):
    def __init__(self, domains, bpe_model_path):  # Remove labels argument
        self.domains = domains
        self.phonetic_preprocessor = PhoneticPreprocessor(vocab_size=512)
        self.phonetic_preprocessor.load_tokenizer(bpe_model_path)
        self.semantic_preprocessor = SemanticPreprocessor()

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        domain = self.domains[idx]

        # Phonetic preprocessing
        tokens, masks = self.phonetic_preprocessor.tokenize_domain(domain, max_length=128)
        phonetic_token_ids = torch.tensor(tokens)

        # Semantic preprocessing
        semantic_embed = self.semantic_preprocessor.tokenize_domain(domain)
        semantic_embed = torch.tensor(semantic_embed)

        return {'phonetic_token_ids': phonetic_token_ids, 'semantic_embed': semantic_embed, 'domain': domain}  # Remove labels

def predict_dga(csv_file, model_path, bpe_model_path, batch_size=32):
    """
    Predicts DGA status for domains in a CSV file using DataLoader for efficiency, assuming all labels are 1.

    Args:
        csv_file (str): Path to the CSV file containing domains. Must have 'domain' and 'subclass' columns.
        model_path (str): Path to the trained DGA detection model (.pth file).
        bpe_model_path (str): Path to the trained BPE model file.
        batch_size (int): Batch size for inference.

    Returns:
        pandas.DataFrame: A DataFrame with the original data plus a 'predicted_isDGA' column (True/False).
    """

    logger.info(f"Starting DGA prediction for file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    df = df[df['subclass'] == 'legit'].copy()
    df = df.dropna(subset=['domain'])

    # nivdort pronounceable = 1
    # bamital pronounceable = 0
    logger.info(f"Total dataset filtered: {len(df)}")
    print(df)

    # Apply wordninja splitting
    logger.info("Applying wordninja splitting to domains...")
    df['domain'] = df['domain'].apply(lambda x: ' '.join(wordninja.split(x)))
    logger.info("Wordninja splitting complete.")

    # Create the Dataset
    domains = df['domain'].tolist()
    inference_dataset = InferenceDataset(domains, bpe_model_path)

    # Create the DataLoader
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    # Load the trained model
    logger.info(f"Loading DGA detection model from: {model_path}")
    model = DGA_Detection_Model(phonetic_embedding_size, semantic_embedding_size, semantic_model)  # Instantiate the model
    model.load_state_dict(torch.load(model_path, map_location=device)) # Load the state dict
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Make predictions
    predicted_labels = []
    with torch.no_grad():
        for batch in inference_dataloader:
            phonetic_token_ids = batch['phonetic_token_ids'].to(device)
            semantic_embeds = batch['semantic_embed'].to(device)

            outputs = model(phonetic_token_ids, semantic_embeds)
            predicted_label_batch = (outputs.squeeze() > 0.5).tolist()  # Convert probabilities to boolean list

            predicted_labels.extend(predicted_label_batch)

    # Add predictions to the DataFrame
    df['predicted_isDGA'] = predicted_labels

    # Calculate Metrics
    true_labels = [0] * len(df)  # All labels are assumed to be 1
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels)

    logger.info(f'Inference Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    logger.info(f'Confusion Matrix:\n{cm}')
    logger.info("DGA prediction completed.")

    return df

if __name__ == '__main__':
    # Example usage:
    csv_file = 'analytics/dga_data-1.csv'  # Replace with your CSV file
    model_path = f'dga_detection_{model_config}.pth'  # Replace with your model file
    bpe_model_path = 'results/512_vocab_size/bpe_models_ipa_full.model'  # Replace with your BPE model file
    batch_size = 32

    try:
        results_df = predict_dga(csv_file, model_path, bpe_model_path, batch_size)
        print(results_df)
        results_df.to_csv('dga_predictions.csv', index=False)
        logger.info("Predictions saved to dga_predictions.csv")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.exception(f"An error occurred during prediction: {e}")