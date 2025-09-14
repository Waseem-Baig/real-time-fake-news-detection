# Real-Time Fake News Detector

A machine learning project to detect fake news articles using NLP techniques and transformer models.

## Project Structure

```
real-time-fake-news-detector/
├── data/              # Dataset files (CSV, JSON)
├── models/            # Trained model files
├── src/               # Source code
│   ├── preprocessing.py   # Data cleaning and preprocessing
│   ├── train_model.py     # Model training scripts
│   └── predict.py         # Prediction functions
├── api/               # Flask API for serving the model
├── notebooks/         # Jupyter notebooks for experimentation
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Download a fake news dataset (e.g., from Kaggle)
3. Run the training script: `python src/train_model.py`
4. Start the API: `python api/app.py`

## Features

- Text preprocessing and cleaning
- TF-IDF + Logistic Regression baseline model
- Optional: Fine-tuned transformer models (BERT, RoBERTa)
- REST API for real-time predictions
- Confidence scores and prediction explanations

## Usage

```python
from src.predict import FakeNewsDetector

detector = FakeNewsDetector()
result = detector.predict("Your news article text here...")
print(f"Prediction: {result['label']}, Confidence: {result['confidence']}")
```

## Next Steps

1. Train baseline model with TF-IDF
2. Experiment with transformer models
3. Deploy as web service
4. Add frontend interface
