"""
Simple transformer-based models using sentence-transformers for fake news detection.
This provides an easier approach to using transformer models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not available. Install with: pip install sentence-transformers")

from dataset_downloader import DatasetDownloader


class SimplifiedTransformerClassifier:
    """
    A simplified transformer-based classifier using sentence embeddings.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', classifier_type='logistic'):
        """
        Initialize the classifier.
        
        Args:
            model_name (str): Name of the sentence transformer model
            classifier_type (str): Type of classifier ('logistic', 'random_forest', 'svm')
        """
        self.model_name = model_name
        self.classifier_type = classifier_type
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers library is required")
        
        print(f"🚀 Loading {model_name} sentence transformer...")
        self.encoder = SentenceTransformer(model_name)
        
        # Initialize classifier
        if classifier_type == 'logistic':
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_type == 'svm':
            self.classifier = SVC(probability=True, random_state=42)
        else:
            raise ValueError("classifier_type must be 'logistic', 'random_forest', or 'svm'")
        
        self.is_trained = False
        print(f"✅ Initialized {classifier_type} classifier with {model_name} embeddings")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts (List[str]): List of texts to encode
            
        Returns:
            np.ndarray: Text embeddings
        """
        print(f"🔄 Encoding {len(texts)} texts...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return embeddings
    
    def train(self, texts: List[str], labels: List[int]):
        """
        Train the classifier.
        
        Args:
            texts (List[str]): Training texts
            labels (List[int]): Training labels
        """
        print(f"🎯 Training {self.classifier_type} classifier...")
        
        # Encode texts
        embeddings = self.encode_texts(texts)
        
        # Train classifier
        self.classifier.fit(embeddings, labels)
        self.is_trained = True
        
        print("✅ Training completed!")
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Make predictions.
        
        Args:
            texts (List[str]): Texts to classify
            
        Returns:
            List[int]: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        embeddings = self.encode_texts(texts)
        return self.classifier.predict(embeddings)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts (List[str]): Texts to classify
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        embeddings = self.encode_texts(texts)
        return self.classifier.predict_proba(embeddings)
    
    def evaluate(self, texts: List[str], labels: List[int]) -> float:
        """
        Evaluate the model.
        
        Args:
            texts (List[str]): Test texts
            labels (List[int]): Test labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(texts)
        accuracy = accuracy_score(labels, predictions)
        
        print(f"\n📊 Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(labels, predictions, target_names=['Fake', 'Real']))
        
        return accuracy
    
    def save(self, filepath: str):
        """
        Save the trained classifier.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        model_data = {
            'classifier': self.classifier,
            'model_name': self.model_name,
            'classifier_type': self.classifier_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load a saved classifier.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            SimplifiedTransformerClassifier: Loaded classifier
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        classifier = cls(
            model_name=model_data['model_name'],
            classifier_type=model_data['classifier_type']
        )
        
        # Load trained classifier
        classifier.classifier = model_data['classifier']
        classifier.is_trained = model_data['is_trained']
        
        print(f"📥 Model loaded from {filepath}")
        return classifier


def train_simple_transformer_models(data_path=None):
    """
    Train simplified transformer models.
    
    Args:
        data_path (str): Path to the dataset
    """
    print("🤖 Training Simplified Transformer Models")
    print("=" * 50)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ sentence-transformers not available!")
        print("💡 Install with: pip install sentence-transformers")
        return {}
    
    # Load dataset
    if data_path and os.path.exists(data_path):
        print(f"📥 Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("📥 Using enhanced sample dataset...")
        data_path = '../data/enhanced_sample_dataset.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            downloader = DatasetDownloader()
            dataset_path = downloader.download_dataset('sample_combined')
            df = pd.read_csv(dataset_path)
    
    print(f"📊 Dataset info:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Fake news: {len(df[df['label'] == 0])}")
    print(f"   - Real news: {len(df[df['label'] == 1])}")
    
    # Prepare data
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Data split:")
    print(f"   - Training: {len(X_train)} samples")
    print(f"   - Testing: {len(X_test)} samples")
    
    # Models to test
    models_config = [
        ('all-MiniLM-L6-v2', 'logistic', 'MiniLM + Logistic'),
        ('all-MiniLM-L6-v2', 'random_forest', 'MiniLM + Random Forest'),
        ('paraphrase-MiniLM-L6-v2', 'logistic', 'Paraphrase-MiniLM + Logistic'),
    ]
    
    results = {}
    trained_models = {}
    
    for model_name, classifier_type, display_name in models_config:
        print(f"\n{'='*20} Training {display_name} {'='*20}")
        
        try:
            # Initialize classifier
            classifier = SimplifiedTransformerClassifier(
                model_name=model_name,
                classifier_type=classifier_type
            )
            
            # Train
            classifier.train(X_train, y_train)
            
            # Evaluate
            accuracy = classifier.evaluate(X_test, y_test)
            results[display_name] = accuracy
            trained_models[display_name] = classifier
            
            # Save model
            model_filename = f"../models/simple_transformer_{model_name.replace('/', '_')}_{classifier_type}.joblib"
            os.makedirs(os.path.dirname(model_filename), exist_ok=True)
            classifier.save(model_filename)
            
            # Sample predictions
            sample_texts = [
                "Scientists discover aliens living on Mars planning invasion",
                "Local government approves new education budget for schools",
                "Miracle cure discovered that reverses aging completely",
                "University researchers publish new climate change study"
            ]
            
            predictions = classifier.predict(sample_texts)
            probabilities = classifier.predict_proba(sample_texts)
            
            print(f"\n🧪 Sample predictions:")
            for i, (text, pred, prob) in enumerate(zip(sample_texts, predictions, probabilities)):
                label = "REAL" if pred == 1 else "FAKE"
                confidence = max(prob) * 100
                print(f"{i+1}. Text: {text[:60]}...")
                print(f"   Prediction: {label} (Confidence: {confidence:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error training {display_name}: {e}")
    
    # Final comparison
    print(f"\n{'='*50}")
    print("📈 Final Results:")
    print("-" * 30)
    
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")
    
    # Find best model
    if results:
        best_model = max(results, key=results.get)
        print(f"\n🏆 Best model: {best_model} ({results[best_model]:.4f})")
    
    return results, trained_models


def install_sentence_transformers():
    """
    Install sentence-transformers if not available.
    """
    print("📦 Installing sentence-transformers...")
    try:
        import subprocess
        import sys
        
        # Get the Python executable path
        python_exe = sys.executable
        
        # Install sentence-transformers
        subprocess.check_call([python_exe, '-m', 'pip', 'install', 'sentence-transformers'])
        
        print("✅ sentence-transformers installed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error installing sentence-transformers: {e}")
        return False


def main():
    """
    Main function to train simplified transformer models.
    """
    print("🚀 Simplified Transformer Models for Fake News Detection")
    print("=" * 60)
    
    # Check if sentence-transformers is available
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("⚠️ sentence-transformers not available!")
        install_choice = input("Would you like to install it now? (y/n): ").strip().lower()
        
        if install_choice == 'y':
            if install_sentence_transformers():
                print("🔄 Please restart the script to use the newly installed library.")
                return
            else:
                print("❌ Installation failed. Please install manually:")
                print("   pip install sentence-transformers")
                return
        else:
            print("❌ Cannot proceed without sentence-transformers")
            return
    
    # Check for enhanced dataset
    data_path = '../data/enhanced_sample_dataset.csv'
    
    # Train models
    results, models = train_simple_transformer_models(data_path)
    
    print(f"\n✅ Training completed!")
    
    if results:
        print(f"\n💡 Next steps:")
        print("1. The best performing model has been saved")
        print("2. You can integrate it into your Flask API")
        print("3. Consider getting a larger dataset for better performance")
        print("4. Try different sentence transformer models for comparison")
    else:
        print("❌ No models were successfully trained")


if __name__ == "__main__":
    main()