# DGA Detection using Deep Learning 

## Overview

This research project presents an approach to detecting Domain Generation Algorithm (DGA) domains by integrating semantic and phonetic features using deep learning techniques.

## Key Features

- **Hybrid Feature Extraction**:
  - Semantic Analysis: Leverages BERT for understanding domain name semantics
  - Phonetic Analysis: Converts domains to phonetic representations (Using BPE Tokenization)
  - Combines multiple feature extraction techniques

- **Deep Learning Architecture**:
  - 1D Convolutional Neural Network (CNN)
  - Self-Attention Mechanism
  - Binary Classification for DGA Detection

## Project Structure

```
dga_detection/
│
├── data/                # Dataset management
│   └── dataset_loader.py
│
├── preprocessing/        # Feature extraction
│   ├── semantic_preprocessor.py
│   └── phonetic_preprocessor.py
│
├── models/               # Neural network model
│   └── dga_detector.py
│
├── utils/                # Utility functions
│   ├── metrics.py
│   └── config.py
│
├── requirements.txt
└── main.py
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/rahmadai/dga-detection.git
cd dga-detection
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to train and evaluate the DGA detection model:

```bash
python main.py
```

## Key Dependencies

- PyTorch
- Transformers (BERT)
- Scikit-learn
- Epitran (Phonetic Conversion)

## Methodology

1. **Data Preparation**
   - Load DGA and benign domain datasets
   - Split into training and testing sets

2. **Feature Extraction**
   - Semantic Features: BERT-based embedding
   - Phonetic Features: BPE-based phonetic embedding

3. **Model Architecture**
   - 1D Convolutional Layer
   - Self-Attention Mechanism
   - Binary Classification Layer

## Performance Metrics

The model evaluates performance using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Limitations and Future Work

- Expand dataset diversity
- Explore additional feature extraction techniques
- Add usage examples

## Citation

If you use this work in your research, please cite:
```
Kurniawan, R. (2024). Detection of Algorithmic Domains Using Deep Learning with Semantic and Phonetic Feature Integration.
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Rahmad Kurniawan
- Email: rahmadkurniawan.ai@gmail.com
```