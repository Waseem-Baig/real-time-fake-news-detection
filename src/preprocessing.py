"""
Data preprocessing utilities for fake news detection.
This module handles text cleaning, tokenization, and feature extraction.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class TextPreprocessor:
    """
    A class for preprocessing text data for fake news detection.
    """
    
    def __init__(self, remove_stopwords=True, use_stemming=False):
        """
        Initialize the preprocessor.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            use_stemming (bool): Whether to apply stemming
        """
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        
        if self.remove_stopwords:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = set()
                
        if self.use_stemming:
            self.stemmer = PorterStemmer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and further process if needed
        if self.remove_stopwords or self.use_stemming:
            try:
                tokens = word_tokenize(text)
                
                # Remove stopwords
                if self.remove_stopwords:
                    tokens = [token for token in tokens if token not in self.stop_words]
                
                # Apply stemming
                if self.use_stemming:
                    tokens = [self.stemmer.stem(token) for token in tokens]
                
                text = ' '.join(tokens)
            except:
                # If NLTK operations fail, return the basic cleaned text
                pass
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts (List[str]): List of texts to preprocess
            
        Returns:
            List[str]: List of preprocessed texts
        """
        return [self.clean_text(text) for text in texts]


def load_and_preprocess_data(file_path: str, text_column: str, label_column: str, 
                           test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and split into train/test sets.
    
    Args:
        file_path (str): Path to the dataset file (CSV)
        text_column (str): Name of the column containing text data
        label_column (str): Name of the column containing labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Remove rows with missing values
    df = df.dropna(subset=[text_column, label_column])
    
    # Split into train and test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, 
        stratify=df[label_column]
    )
    
    return train_df, test_df


def preprocess_dataset(df: pd.DataFrame, text_column: str, 
                      preprocessor: TextPreprocessor = None) -> pd.DataFrame:
    """
    Preprocess an entire dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        text_column (str): Name of the column containing text data
        preprocessor (TextPreprocessor): Preprocessor instance to use
        
    Returns:
        pd.DataFrame: DataFrame with preprocessed text
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    df_copy = df.copy()
    df_copy[text_column] = preprocessor.preprocess_batch(df[text_column].tolist())
    
    return df_copy


# Example usage and testing
if __name__ == "__main__":
    # Create a sample preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True, use_stemming=False)
    
    # Test with sample text
    sample_text = """
    <h1>Breaking News!</h1> 
    This is a FAKE news article from https://fakenews.com with some @email@example.com!!! 
    It contains HTML tags, URLs, and special characters!!!
    """
    
    cleaned_text = preprocessor.clean_text(sample_text)
    print("Original text:")
    print(sample_text)
    print("\nCleaned text:")
    print(cleaned_text)
    
    # Test batch processing
    texts = [
        "This is the first news article with some noise!",
        "Second article with URLs: https://example.com",
        "Third article with email@test.com and numbers 123"
    ]
    
    cleaned_texts = preprocessor.preprocess_batch(texts)
    print("\nBatch processing results:")
    for i, (original, cleaned) in enumerate(zip(texts, cleaned_texts)):
        print(f"{i+1}. Original: {original}")
        print(f"   Cleaned:  {cleaned}")