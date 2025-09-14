"""
Model wrapper to load and use the traditional TF-IDF models with the Flask API.
"""

import joblib
import os
import numpy as np
from typing import List, Tuple

class TraditionalModelWrapper:
    """Wrapper class for traditional TF-IDF models."""
    
    def __init__(self, model_path: str, vectorizer_path: str, preprocessor_path: str):
        """Initialize the wrapper with model components."""
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.preprocessor = joblib.load(preprocessor_path)
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on a list of texts."""
        # Preprocess texts
        processed_texts = []
        for text in texts:
            processed_text = self.preprocessor.clean_text(text)
            processed_texts.append(processed_text)
        
        # Vectorize
        X = self.vectorizer.transform(processed_texts)
        
        # Predict
        return self.model.predict(X)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        # Preprocess texts
        processed_texts = []
        for text in texts:
            processed_text = self.preprocessor.clean_text(text)
            processed_texts.append(processed_text)
        
        # Vectorize
        X = self.vectorizer.transform(processed_texts)
        
        # Get probabilities
        return self.model.predict_proba(X)

def load_traditional_model(model_name: str, models_dir: str) -> TraditionalModelWrapper:
    """Load a traditional model by name."""
    if model_name == 'tfidf_logistic':
        model_path = os.path.join(models_dir, 'fake_news_logistic.joblib')
        vectorizer_path = os.path.join(models_dir, 'vectorizer_logistic.joblib')
        preprocessor_path = os.path.join(models_dir, 'preprocessor_logistic.joblib')
    elif model_name == 'tfidf_random_forest':
        model_path = os.path.join(models_dir, 'fake_news_random_forest.joblib')
        vectorizer_path = os.path.join(models_dir, 'vectorizer_random_forest.joblib')
        preprocessor_path = os.path.join(models_dir, 'preprocessor_random_forest.joblib')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Check if all files exist
    for path in [model_path, vectorizer_path, preprocessor_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
    
    return TraditionalModelWrapper(model_path, vectorizer_path, preprocessor_path)