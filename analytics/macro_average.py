import pandas as pd
import numpy as np

data = pd.read_csv("macro-average.csv")

def calculate_averages(df):
    # Macro Average - simple mean of metrics
    macro_precision = df['Precision'].mean()
    macro_recall = df['Recall'].mean()
    macro_f1 = df['F1-Score'].mean()
    
    # Micro Average - weighted by test data size
    total_samples = df['Test Data'].sum()
    micro_precision = (df['Precision'] * df['Test Data']).sum() / total_samples
    micro_recall = (df['Recall'] * df['Test Data']).sum() / total_samples
    micro_f1 = (df['F1-Score'] * df['Test Data']).sum() / total_samples
    
    return {
        'Macro': {
            'Precision': macro_precision,
            'Recall': macro_recall,
            'F1-Score': macro_f1
        },
        'Micro': {
            'Precision': micro_precision,
            'Recall': micro_recall,
            'F1-Score': micro_f1
        }
    }

# Calculate averages
results = calculate_averages(data)

# Print results in a formatted way
print("Macro Averages:")
print(f"Precision: {results['Macro']['Precision']:.4f}")
print(f"Recall: {results['Macro']['Recall']:.4f}")
print(f"F1-Score: {results['Macro']['F1-Score']:.4f}")
print("\nMicro Averages:")
print(f"Precision: {results['Micro']['Precision']:.4f}")
print(f"Recall: {results['Micro']['Recall']:.4f}")
print(f"F1-Score: {results['Micro']['F1-Score']:.4f}")