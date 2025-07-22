# NCCN Lung Cancer Sentence Classifier

_AIPI 540 – Deep Learning (LLM) – Duke University_  
**Author:** Evan Moh

## Overview

This project builds automated text classification models to distinguish between sentences about **Small Cell Lung Cancer (SCLC)** and **Non-Small Cell Lung Cancer (NSCLC)** using NCCN guideline text.  
It compares naive keyword matching, classical machine learning, and deep learning.  
It also provides a Streamlit web app for clinical text classification.

## File Structures
├── app.py                  # Streamlit web application
├── data_prep.py            # Data cleaning, PDF extraction, and data splitting
├── ml.py                   # Classical ML model (TF-IDF + Logistic Regression)
├── nn.py                   # Neural network (Keras) training and evaluation
├── naive.py                # Naive keyword search baseline
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│   │   ├── all.csv         # All cleaned & labeled sentences
│   │   ├── train.csv       # Training split (70%)
│   │   └── test.csv        # Test split (30%)
│   └── raw/
│       ├── nsclc_nccn.pdf  # NSCLC guideline PDF
│       └── sclc_nccn.pdf   # SCLC guideline PDF
├── models/
│   ├── nn_model.keras      # Trained Keras neural network model
│   ├── nn_tokenizer.pkl    # Tokenizer for neural net


## Data Preparation

- Extract sentences from NCCN PDFs for NSCLC and SCLC using `data_prep.py`
- Clean, filter, and label sentences based on guideline source
- Deduplicate and split into **train (70%)** and **test (30%)** CSVs

## Modeling Approaches

### Naive Keyword Search (`naive.py`)
- Predicts by checking SCLC/NSCLC keyword counts in each sentence
- Simple, transparent baseline

### Classical Machine Learning (`ml.py`)
- Preprocessing: lowercasing, lemmatization, stopword/punctuation removal
- Features: TF-IDF (unigram + bigram)
- Classifier: Logistic Regression
- Good overall accuracy, but lower recall for SCLC

### Deep Learning Neural Network (`nn.py`)
- Tokenize and pad sentences
- Keras Sequential: Embedding → GlobalAveragePooling1D → Dense (ReLU) → Dropout → Dense (Sigmoid)
- Early stopping to avoid overfitting
- Best SCLC and overall performance

## Streamlit Web App

- `app.py` lets users classify new clinical sentences (NSCLC or SCLC) and see confidence
- Paste text or use built-in clinical examples
- Uses the neural net model + tokenizer

### To run the app:

streamlit run app.py

## Results

- **NeuralNet:** Highest accuracy (0.87) and SCLC F1 (0.68); best balance
- **Classical ML:** High accuracy, lower SCLC recall/F1
- **Naive:** Good for NSCLC, poor for SCLC

**Conclusion:**  
_The neural net model is recommended for its balanced and reliable performance._

## Installation

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Download spaCy model:
    ```
    python -m spacy download en_core_web_sm
    ```

## Usage

- Data prep:  
  Run `data_prep.py` to process NCCN PDFs
- Model training:  
  Run `nn.py` or `ml.py` to train/evaluate models
- App:  
  Run `streamlit run app.py` to use the classifier

## Ethics Statement

- Uses only public NCCN guideline text
- No patient data, no PHI or confidential info
- For academic/educational use only

## License

See `LICENSE`.

