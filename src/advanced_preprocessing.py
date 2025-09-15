"""
Advanced preprocessing utilities for enhanced fake news detection.
This module implements state-of-the-art text preprocessing techniques.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
for data in required_nltk_data:
    try:
        nltk.download(data, quiet=True)
    except:
        pass

class AdvancedTextPreprocessor:
    """
    Advanced text preprocessor with enhanced feature extraction capabilities.
    """
    
    def __init__(self, 
                 remove_stopwords=True, 
                 use_lemmatization=True,
                 extract_features=True,
                 min_word_length=2,
                 max_word_length=20):
        """
        Initialize the advanced preprocessor.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            use_lemmatization (bool): Whether to use lemmatization
            extract_features (bool): Whether to extract linguistic features
            min_word_length (int): Minimum word length to keep
            max_word_length (int): Maximum word length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.extract_features = extract_features
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        
        # Initialize NLTK components
        if self.remove_stopwords:
            try:
                self.stop_words = set(stopwords.words('english'))
                # Add custom stopwords for news domain
                custom_stops = {'said', 'says', 'according', 'reported', 'news', 'report'}
                self.stop_words.update(custom_stops)
            except:
                self.stop_words = set()
                
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
            
        # Initialize sentiment analyzer
        if self.extract_features:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatizer accepts."""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def clean_text(self, text: str) -> str:
        """
        Advanced text cleaning with multiple techniques.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        
        # Remove numbers (optional - you might want to keep them)
        text = re.sub(r'\d+', ' ', text)
        
        return text.strip()
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        try:
            words = word_tokenize(text)
        except:
            words = text.split()
        
        # Filter words by length
        words = [word for word in words if self.min_word_length <= len(word) <= self.max_word_length]
        
        # Remove stopwords
        if self.remove_stopwords:
            words = [word for word in words if word.lower() not in self.stop_words]
        
        # Lemmatization
        if self.use_lemmatization:
            words = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in words]
        
        return ' '.join(words)
    
    def extract_linguistic_features(self, text: str) -> Dict:
        """
        Extract advanced linguistic features from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Dictionary of extracted features
        """
        features = {}
        
        if not isinstance(text, str) or len(text) == 0:
            return {
                'word_count': 0, 'char_count': 0, 'sentence_count': 0,
                'avg_word_length': 0, 'avg_sentence_length': 0,
                'exclamation_count': 0, 'question_count': 0,
                'upper_case_ratio': 0, 'readability_score': 0,
                'sentiment_positive': 0, 'sentiment_negative': 0,
                'sentiment_neutral': 0, 'sentiment_compound': 0
            }
        
        # Basic text statistics
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # Case features
        features['upper_case_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Readability score
        try:
            features['readability_score'] = textstat.flesch_reading_ease(text)
        except:
            features['readability_score'] = 0
        
        # Sentiment analysis
        if self.extract_features:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
            features['sentiment_positive'] = sentiment_scores['pos']
            features['sentiment_negative'] = sentiment_scores['neg']
            features['sentiment_neutral'] = sentiment_scores['neu']
            features['sentiment_compound'] = sentiment_scores['compound']
        
        return features
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[str]: List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
    
    def extract_features_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract features from a batch of texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        features_list = []
        for text in texts:
            features = self.extract_linguistic_features(text)
            features_list.append(features)
        
        return pd.DataFrame(features_list)


class EnhancedTfidfVectorizer:
    """
    Enhanced TF-IDF vectorizer with advanced features.
    """
    
    def __init__(self,
                 max_features=50000,
                 ngram_range=(1, 3),
                 min_df=2,
                 max_df=0.95,
                 sublinear_tf=True,
                 use_feature_selection=True,
                 k_best_features=10000):
        """
        Initialize enhanced TF-IDF vectorizer.
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
            sublinear_tf (bool): Apply sublinear TF scaling
            use_feature_selection (bool): Use feature selection
            k_best_features (int): Number of best features to select
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            stop_words='english'
        )
        
        self.use_feature_selection = use_feature_selection
        self.k_best_features = k_best_features
        
        if self.use_feature_selection:
            self.feature_selector = SelectKBest(chi2, k=k_best_features)
    
    def fit_transform(self, texts, labels=None):
        """
        Fit vectorizer and transform texts.
        
        Args:
            texts: Input texts
            labels: Labels for feature selection
            
        Returns:
            Transformed features
        """
        # Fit and transform with TF-IDF
        X = self.vectorizer.fit_transform(texts)
        
        # Apply feature selection if enabled
        if self.use_feature_selection and labels is not None:
            X = self.feature_selector.fit_transform(X, labels)
        
        return X
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: Input texts
            
        Returns:
            Transformed features
        """
        X = self.vectorizer.transform(texts)
        
        if self.use_feature_selection:
            X = self.feature_selector.transform(X)
        
        return X


def preprocess_dataset_advanced(df: pd.DataFrame, 
                               text_column='text', 
                               label_column='label',
                               test_size=0.2,
                               random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Advanced preprocessing of the entire dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of text column
        label_column (str): Name of label column
        test_size (float): Test set size
        random_state (int): Random state for reproducibility
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    # Initialize preprocessor
    preprocessor = AdvancedTextPreprocessor()
    
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Preprocess texts
    print("Preprocessing texts...")
    df_processed['processed_text'] = preprocessor.preprocess_batch(df_processed[text_column].tolist())
    
    # Extract linguistic features
    print("Extracting linguistic features...")
    features_df = preprocessor.extract_features_batch(df_processed[text_column].tolist())
    
    # Combine with original data
    df_processed = pd.concat([df_processed, features_df], axis=1)
    
    # Split into train and test
    train_df, test_df = train_test_split(
        df_processed, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df_processed[label_column]
    )
    
    print(f"Dataset split: {len(train_df)} train, {len(test_df)} test samples")
    
    return train_df, test_df