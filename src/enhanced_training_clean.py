"""
Enhanced training script with advanced models and optimization techniques.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from advanced_preprocessing import AdvancedTextPreprocessor, EnhancedTfidfVectorizer


class EnhancedFakeNewsClassifier:
    """Enhanced machine learning classifier with advanced techniques."""
    
    def __init__(self, 
                 model_type='ensemble',
                 use_advanced_preprocessing=True,
                 use_feature_engineering=True,
                 hyperparameter_tuning=True):
        """Initialize the enhanced classifier."""
        self.model_type = model_type
        self.use_advanced_preprocessing = use_advanced_preprocessing
        self.use_feature_engineering = use_feature_engineering
        self.hyperparameter_tuning = hyperparameter_tuning
        
        # Initialize components
        if self.use_advanced_preprocessing:
            self.preprocessor = AdvancedTextPreprocessor()
            self.vectorizer = EnhancedTfidfVectorizer()
        else:
            self.preprocessor = None
            self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        
        # Initialize model
        self.model = self._get_model()
        self.is_trained = False
    
    def _get_model(self):
        """Initialize the appropriate model based on model_type."""
        if self.model_type == 'logistic':
            return LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200, 
                max_depth=20, 
                random_state=42, 
                class_weight='balanced',
                n_jobs=-1
            )
        
        elif self.model_type == 'ensemble':
            # Create ensemble of different models
            models = [
                ('lr', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
            ]
            return VotingClassifier(models, voting='soft')
        
        else:
            raise ValueError("model_type must be one of: 'logistic', 'random_forest', 'ensemble'")
    
    def _get_hyperparameter_grid(self):
        """Get hyperparameter grid for tuning."""
        if self.model_type == 'logistic':
            return {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        return {}
    
    def preprocess_text(self, texts):
        """Preprocess texts if preprocessing is enabled."""
        if self.preprocessor:
            return self.preprocessor.preprocess_batch(texts)
        return texts
    
    def train(self, train_df, val_df=None, text_column='text', label_column='label'):
        """Train the model with advanced techniques."""
        print(f"Training {self.model_type} model...")
        
        # Extract texts and labels
        X_train = train_df[text_column].tolist()
        y_train = train_df[label_column].tolist()
        
        # Preprocess texts
        if self.use_advanced_preprocessing:
            X_train_processed = self.preprocess_text(X_train)
            # Vectorize with feature selection
            X_train_vec = self.vectorizer.fit_transform(X_train_processed, y_train)
        else:
            X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Add engineered features if enabled
        if self.use_feature_engineering and self.use_advanced_preprocessing:
            # Get feature columns (exclude text columns)
            feature_columns = [col for col in train_df.columns 
                             if col not in [text_column, label_column, 'processed_text']]
            
            if feature_columns:
                X_features = train_df[feature_columns].values
                # Scale features
                if not hasattr(self, 'scaler'):
                    self.scaler = StandardScaler()
                    X_features_scaled = self.scaler.fit_transform(X_features)
                else:
                    X_features_scaled = self.scaler.transform(X_features)
                
                # Combine text and engineered features
                from scipy.sparse import hstack
                X_train_vec = hstack([X_train_vec, X_features_scaled])
        
        # Hyperparameter tuning
        if self.hyperparameter_tuning and self.model_type != 'ensemble':
            print("Performing hyperparameter tuning...")
            param_grid = self._get_hyperparameter_grid()
            
            if param_grid:
                grid_search = GridSearchCV(
                    self.model, 
                    param_grid, 
                    cv=3, 
                    scoring='f1_weighted', 
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train_vec, y_train)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True
        
        # Print training results
        train_pred = self.model.predict(X_train_vec)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Training F1-Score: {train_f1:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_vec, y_train, cv=5, scoring='f1_weighted')
        print(f"Cross-validation F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Validate if validation data is provided
        if val_df is not None:
            val_accuracy, val_f1 = self.evaluate(val_df, text_column, label_column)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation F1-Score: {val_f1:.4f}")
    
    def predict(self, texts):
        """Make predictions on new texts."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess texts
        if self.use_advanced_preprocessing:
            texts_processed = self.preprocess_text(texts)
            texts_vec = self.vectorizer.transform(texts_processed)
        else:
            texts_vec = self.vectorizer.transform(texts)
        
        return self.model.predict(texts_vec)
    
    def predict_proba(self, texts):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess texts
        if self.use_advanced_preprocessing:
            texts_processed = self.preprocess_text(texts)
            texts_vec = self.vectorizer.transform(texts_processed)
        else:
            texts_vec = self.vectorizer.transform(texts)
        
        return self.model.predict_proba(texts_vec)
    
    def evaluate(self, test_df, text_column='text', label_column='label'):
        """Evaluate the model on test data."""
        X_test = test_df[text_column].tolist()
        y_test = test_df[label_column].tolist()
        
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        print(f"\\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"\\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        return accuracy, f1
    
    def save_model(self, model_dir='../models'):
        """Save the trained model and components."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f'{self.model_type}_enhanced_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save vectorizer
        vectorizer_path = os.path.join(model_dir, f'{self.model_type}_enhanced_vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save preprocessor if used
        if self.preprocessor:
            preprocessor_path = os.path.join(model_dir, f'{self.model_type}_enhanced_preprocessor.pkl')
            joblib.dump(self.preprocessor, preprocessor_path)
        
        # Save scaler if used
        if hasattr(self, 'scaler'):
            scaler_path = os.path.join(model_dir, f'{self.model_type}_enhanced_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_dir}")


def train_multiple_models(train_df, val_df=None, text_column='text', label_column='label'):
    """Train multiple models and compare their performance."""
    model_types = ['logistic', 'random_forest', 'ensemble']
    results = {}
    
    for model_type in model_types:
        print(f"\\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        try:
            # Initialize classifier
            classifier = EnhancedFakeNewsClassifier(
                model_type=model_type,
                use_advanced_preprocessing=True,
                use_feature_engineering=True,
                hyperparameter_tuning=True
            )
            
            # Train model
            classifier.train(train_df, val_df, text_column, label_column)
            
            # Evaluate on validation set if provided
            if val_df is not None:
                val_accuracy, val_f1 = classifier.evaluate(val_df, text_column, label_column)
                results[model_type] = {
                    'model': classifier,
                    'val_accuracy': val_accuracy,
                    'val_f1': val_f1
                }
            else:
                results[model_type] = {'model': classifier}
            
            # Save model
            classifier.save_model()
            
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            continue
    
    return results


if __name__ == "__main__":
    print("Enhanced Fake News Detection Training")
    print("=====================================")
    
    # This would be used with your actual dataset
    # Example: df = pd.read_csv('../data/fake_news_dataset.csv')
    print("Ready to train enhanced models!")
    print("Use train_multiple_models() function to train and compare models.")