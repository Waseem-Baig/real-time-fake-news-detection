"""
Training script for fake news detection model.
This module implements both baseline and advanced models.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from preprocessing import TextPreprocessor, preprocess_dataset


class FakeNewsClassifier:
    """
    A machine learning classifier for fake news detection.
    """
    
    def __init__(self, model_type='logistic', use_preprocessing=True):
        """
        Initialize the classifier.
        
        Args:
            model_type (str): Type of model ('logistic' or 'random_forest')
            use_preprocessing (bool): Whether to use text preprocessing
        """
        self.model_type = model_type
        self.use_preprocessing = use_preprocessing
        
        # Initialize components
        self.preprocessor = TextPreprocessor() if use_preprocessing else None
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english' if not use_preprocessing else None
        )
        
        # Initialize model
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")
    
    def preprocess_text(self, texts):
        """Preprocess texts if preprocessing is enabled."""
        if self.preprocessor:
            return self.preprocessor.preprocess_batch(texts)
        return texts
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on training data.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data (optional)
            y_val: Validation labels (optional)
        """
        # Preprocess texts
        X_train_processed = self.preprocess_text(X_train)
        
        # Vectorize texts
        X_train_vec = self.vectorizer.fit_transform(X_train_processed)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        
        # Print training accuracy
        train_pred = self.model.predict(X_train_vec)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Validate if validation data is provided
        if X_val is not None and y_val is not None:
            val_accuracy = self.evaluate(X_val, y_val)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    def predict(self, texts):
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            Predictions (0 for fake, 1 for real)
        """
        # Preprocess texts
        texts_processed = self.preprocess_text(texts)
        
        # Vectorize texts
        texts_vec = self.vectorizer.transform(texts_processed)
        
        # Make predictions
        return self.model.predict(texts_vec)
    
    def predict_proba(self, texts):
        """
        Get prediction probabilities.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            Probability scores for each class
        """
        # Preprocess texts
        texts_processed = self.preprocess_text(texts)
        
        # Vectorize texts
        texts_vec = self.vectorizer.transform(texts_processed)
        
        # Get probabilities
        return self.model.predict_proba(texts_vec)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test text data
            y_test: Test labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Print detailed evaluation
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        return accuracy
    
    def save_model(self, model_dir='../models'):
        """Save the trained model and vectorizer."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f'fake_news_{self.model_type}.joblib')
        joblib.dump(self.model, model_path)
        
        # Save vectorizer
        vectorizer_path = os.path.join(model_dir, f'vectorizer_{self.model_type}.joblib')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save preprocessor if used
        if self.preprocessor:
            preprocessor_path = os.path.join(model_dir, f'preprocessor_{self.model_type}.joblib')
            joblib.dump(self.preprocessor, preprocessor_path)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_dir='../models'):
        """Load a previously trained model."""
        # Load model
        model_path = os.path.join(model_dir, f'fake_news_{self.model_type}.joblib')
        self.model = joblib.load(model_path)
        
        # Load vectorizer
        vectorizer_path = os.path.join(model_dir, f'vectorizer_{self.model_type}.joblib')
        self.vectorizer = joblib.load(vectorizer_path)
        
        # Load preprocessor if it exists
        if self.use_preprocessing:
            preprocessor_path = os.path.join(model_dir, f'preprocessor_{self.model_type}.joblib')
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
        
        print(f"Model loaded from {model_path}")


def create_sample_dataset():
    """
    Create a sample dataset for testing (when no real dataset is available).
    """
    # Sample fake news examples
    fake_news = [
        "Scientists have discovered that aliens are living among us and controlling the government",
        "Breaking: Eating chocolate cures all diseases according to secret government study",
        "Local man discovers miracle weight loss pill that doctors don't want you to know about",
        "New research shows vaccines contain microchips for mind control",
        "Celebrity spotted with three-headed alien baby in grocery store",
        "Government admits to hiding cure for aging for the past 50 years",
        "Breaking: Time travel machine invented by teenager in garage",
        "Doctors hate this one weird trick that cures everything",
        "Secret society of billionaires controls all world governments",
        "Eating this common food can give you superpowers, studies show"
    ]
    
    # Sample real news examples
    real_news = [
        "Stock market closes up 2% following positive earnings reports from tech companies",
        "Local university receives grant to study climate change effects on coastal regions",
        "City council approves new budget for infrastructure improvements next year",
        "Health officials recommend annual flu vaccination for all adults",
        "New study shows benefits of regular exercise for mental health",
        "Technology company announces plans to expand operations and hire 500 new employees",
        "Education department launches program to improve literacy rates in rural areas",
        "Researchers develop new treatment approach for diabetes management",
        "Transportation authority announces schedule changes for public bus routes",
        "Environmental group partners with local businesses to reduce plastic waste"
    ]
    
    # Create DataFrame
    data = []
    for text in fake_news:
        data.append({'text': text, 'label': 0})  # 0 for fake
    for text in real_news:
        data.append({'text': text, 'label': 1})  # 1 for real
    
    df = pd.DataFrame(data)
    return df


def train_baseline_model(data_path=None):
    """
    Train a baseline fake news detection model.
    
    Args:
        data_path (str): Path to dataset CSV file. If None, uses sample data.
    """
    print("Training Fake News Detection Model...")
    print("=" * 50)
    
    # Load or create dataset
    if data_path and os.path.exists(data_path):
        print(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        # Assume the CSV has 'text' and 'label' columns
        # Adjust column names as needed for your dataset
    else:
        print("Using sample dataset for demonstration...")
        df = create_sample_dataset()
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    
    # Split data
    X = df['text'].tolist()
    y = df['label'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train both models
    models = ['logistic', 'random_forest']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*20} Training {model_type.upper()} Model {'='*20}")
        
        # Initialize and train classifier
        classifier = FakeNewsClassifier(model_type=model_type, use_preprocessing=True)
        classifier.train(X_train, y_train)
        
        # Evaluate
        accuracy = classifier.evaluate(X_test, y_test)
        results[model_type] = accuracy
        
        # Save model
        classifier.save_model()
        
        # Test with sample predictions
        sample_texts = [
            "Scientists discover aliens living on Mars",
            "Local government approves new education budget"
        ]
        predictions = classifier.predict(sample_texts)
        probabilities = classifier.predict_proba(sample_texts)
        
        print(f"\nSample Predictions:")
        for i, (text, pred, prob) in enumerate(zip(sample_texts, predictions, probabilities)):
            label = "REAL" if pred == 1 else "FAKE"
            confidence = max(prob) * 100
            print(f"{i+1}. Text: {text}")
            print(f"   Prediction: {label} (Confidence: {confidence:.1f}%)")
    
    print(f"\n{'='*50}")
    print("Final Results:")
    for model_type, accuracy in results.items():
        print(f"{model_type.upper()}: {accuracy:.4f}")
    
    return results


if __name__ == "__main__":
    # Train the baseline model
    results = train_baseline_model()
    
    print("\nTraining completed! You can now use the trained models for predictions.")
    print("Next steps:")
    print("1. Get a real dataset (e.g., from Kaggle)")
    print("2. Update the data_path in train_baseline_model()")
    print("3. Set up the Flask API for real-time predictions")