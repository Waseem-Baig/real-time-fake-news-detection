"""
Advanced feature extraction for fake news detection.
This module implements sentiment analysis, source credibility, and other advanced features.
"""

import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️ textblob not available. Install with: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️ vaderSentiment not available. Install with: pip install vaderSentiment")

import urllib.parse
from collections import Counter


class AdvancedFeatureExtractor:
    """
    Extract advanced features from news articles for better fake news detection.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        # Initialize sentiment analyzers
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Known credible sources (can be expanded)
        self.credible_sources = {
            'reuters.com', 'ap.org', 'bbc.com', 'cnn.com', 'npr.org',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com',
            'abcnews.go.com', 'cbsnews.com', 'nbcnews.com', 'pbs.org',
            'time.com', 'newsweek.com', 'usatoday.com', 'wsj.com',
            'bloomberg.com', 'economist.com', 'politico.com'
        }
        
        # Known unreliable sources (can be expanded)
        self.unreliable_sources = {
            'infowars.com', 'breitbart.com', 'naturalnews.com',
            'beforeitsnews.com', 'worldnewsdailyreport.com',
            'theonion.com', 'clickhole.com'  # Satire sites
        }
        
        # Fake news indicators (keywords/phrases)
        self.fake_indicators = [
            'breaking', 'shocking', 'unbelievable', 'miracle', 'secret',
            'doctors hate', 'governments don\'t want you to know',
            'click here', 'you won\'t believe', 'amazing discovery',
            'conspiracy', 'cover-up', 'hidden truth', 'leaked',
            'exclusive', 'banned', 'censored', 'suppressed'
        ]
        
        # Real news indicators
        self.real_indicators = [
            'according to', 'study shows', 'research indicates',
            'experts say', 'data suggests', 'reported by',
            'official statement', 'press release', 'conference',
            'spokesperson', 'analysis', 'investigation'
        ]
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment-based features.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Sentiment features
        """
        features = {}
        
        # TextBlob sentiment
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                features['textblob_polarity'] = blob.sentiment.polarity
                features['textblob_subjectivity'] = blob.sentiment.subjectivity
            except:
                features['textblob_polarity'] = 0.0
                features['textblob_subjectivity'] = 0.0
        else:
            features['textblob_polarity'] = 0.0
            features['textblob_subjectivity'] = 0.0
        
        # VADER sentiment
        if VADER_AVAILABLE:
            try:
                scores = self.vader_analyzer.polarity_scores(text)
                features['vader_positive'] = scores['pos']
                features['vader_negative'] = scores['neg']
                features['vader_neutral'] = scores['neu']
                features['vader_compound'] = scores['compound']
            except:
                features['vader_positive'] = 0.0
                features['vader_negative'] = 0.0
                features['vader_neutral'] = 1.0
                features['vader_compound'] = 0.0
        else:
            features['vader_positive'] = 0.0
            features['vader_negative'] = 0.0
            features['vader_neutral'] = 1.0
            features['vader_compound'] = 0.0
        
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Linguistic features
        """
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        
        # Capitalization patterns
        words = text.split()
        if words:
            features['all_caps_words'] = sum(1 for word in words if word.isupper() and len(word) > 1)
            features['title_case_ratio'] = sum(1 for word in words if word.istitle()) / len(words)
        else:
            features['all_caps_words'] = 0
            features['title_case_ratio'] = 0
        
        return features
    
    def extract_credibility_features(self, text: str, source: str = None, url: str = None) -> Dict[str, float]:
        """
        Extract source credibility features.
        
        Args:
            text (str): Article text
            source (str): Source name
            url (str): Article URL
            
        Returns:
            Dict[str, float]: Credibility features
        """
        features = {}
        
        # Source credibility
        if source:
            source_lower = source.lower()
            features['is_credible_source'] = 1.0 if any(cred in source_lower for cred in self.credible_sources) else 0.0
            features['is_unreliable_source'] = 1.0 if any(unrel in source_lower for unrel in self.unreliable_sources) else 0.0
        else:
            features['is_credible_source'] = 0.0
            features['is_unreliable_source'] = 0.0
        
        # URL analysis
        if url:
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc.lower()
            
            features['domain_credibility'] = 1.0 if domain in self.credible_sources else 0.0
            features['domain_unreliability'] = 1.0 if domain in self.unreliable_sources else 0.0
            features['has_https'] = 1.0 if parsed_url.scheme == 'https' else 0.0
            features['url_length'] = len(url)
        else:
            features['domain_credibility'] = 0.0
            features['domain_unreliability'] = 0.0
            features['has_https'] = 0.0
            features['url_length'] = 0
        
        # Content credibility indicators
        text_lower = text.lower()
        
        # Fake news indicators
        fake_score = sum(1 for indicator in self.fake_indicators if indicator in text_lower)
        features['fake_indicators_count'] = fake_score
        features['fake_indicators_ratio'] = fake_score / max(len(text.split()), 1)
        
        # Real news indicators
        real_score = sum(1 for indicator in self.real_indicators if indicator in text_lower)
        features['real_indicators_count'] = real_score
        features['real_indicators_ratio'] = real_score / max(len(text.split()), 1)
        
        # Citation patterns
        features['has_quotes'] = 1.0 if '"' in text or "'" in text else 0.0
        features['has_attribution'] = 1.0 if any(attr in text_lower for attr in ['said', 'stated', 'according to', 'reported']) else 0.0
        features['has_numbers'] = 1.0 if re.search(r'\d+', text) else 0.0
        
        return features
    
    def extract_emotional_features(self, text: str) -> Dict[str, float]:
        """
        Extract emotional manipulation features.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Emotional features
        """
        features = {}
        
        # Emotional words
        emotional_words = [
            'shocking', 'amazing', 'incredible', 'unbelievable', 'devastating',
            'terrifying', 'horrific', 'outrageous', 'scandalous', 'explosive'
        ]
        
        urgency_words = [
            'urgent', 'immediate', 'breaking', 'alert', 'warning',
            'emergency', 'critical', 'now', 'today', 'quickly'
        ]
        
        text_lower = text.lower()
        
        # Count emotional and urgency words
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        
        features['emotional_words_count'] = emotional_count
        features['urgency_words_count'] = urgency_count
        features['emotional_intensity'] = (emotional_count + urgency_count) / max(len(text.split()), 1)
        
        # Clickbait patterns
        clickbait_patterns = [
            r"you won't believe",
            r"this will shock you",
            r"number \d+ will",
            r"\d+ things",
            r"click here",
            r"find out",
            r"the truth about"
        ]
        
        clickbait_score = sum(1 for pattern in clickbait_patterns if re.search(pattern, text_lower))
        features['clickbait_score'] = clickbait_score
        
        return features
    
    def extract_all_features(self, text: str, source: str = None, url: str = None, title: str = None) -> Dict[str, float]:
        """
        Extract all advanced features from a news article.
        
        Args:
            text (str): Article text
            source (str): Source name
            url (str): Article URL
            title (str): Article title
            
        Returns:
            Dict[str, float]: All extracted features
        """
        features = {}
        
        # Combine title and text for analysis
        full_text = f"{title} {text}" if title else text
        
        # Extract different types of features
        features.update(self.extract_sentiment_features(full_text))
        features.update(self.extract_linguistic_features(full_text))
        features.update(self.extract_credibility_features(full_text, source, url))
        features.update(self.extract_emotional_features(full_text))
        
        return features
    
    def extract_features_batch(self, articles: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract features for a batch of articles.
        
        Args:
            articles: List of article dictionaries with keys: 'text', 'source', 'url', 'title'
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        all_features = []
        
        for article in articles:
            text = article.get('text', '')
            source = article.get('source', None)
            url = article.get('url', None)
            title = article.get('title', None)
            
            features = self.extract_all_features(text, source, url, title)
            all_features.append(features)
        
        return pd.DataFrame(all_features)


class EnhancedFakeNewsClassifier:
    """
    Enhanced classifier that uses advanced features along with traditional text features.
    """
    
    def __init__(self, base_classifier=None):
        """
        Initialize the enhanced classifier.
        
        Args:
            base_classifier: Base text classifier to use
        """
        self.base_classifier = base_classifier
        self.feature_extractor = AdvancedFeatureExtractor()
        self.feature_columns = None
        self.is_trained = False
    
    def prepare_features(self, articles: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training or prediction.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Tuple of text embeddings and advanced features
        """
        # Extract advanced features
        advanced_features_df = self.feature_extractor.extract_features_batch(articles)
        
        # Get text for base classifier
        texts = [article.get('text', '') for article in articles]
        
        # Get text embeddings if base classifier is available
        if self.base_classifier and hasattr(self.base_classifier, 'encode_texts'):
            text_embeddings = self.base_classifier.encode_texts(texts)
        else:
            # Create dummy embeddings if no base classifier
            text_embeddings = np.zeros((len(texts), 1))
        
        # Store feature columns for later use
        if self.feature_columns is None:
            self.feature_columns = advanced_features_df.columns.tolist()
        
        # Ensure consistent feature columns
        for col in self.feature_columns:
            if col not in advanced_features_df.columns:
                advanced_features_df[col] = 0.0
        
        advanced_features_df = advanced_features_df[self.feature_columns]
        
        return text_embeddings, advanced_features_df.values
    
    def predict_enhanced(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make enhanced predictions using both text and advanced features.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of prediction results with detailed analysis
        """
        results = []
        
        for article in articles:
            # Get basic text prediction if base classifier available
            if self.base_classifier and hasattr(self.base_classifier, 'predict'):
                try:
                    base_pred = self.base_classifier.predict([article.get('text', '')])
                    base_prob = self.base_classifier.predict_proba([article.get('text', '')])
                    base_prediction = base_pred[0]
                    base_confidence = max(base_prob[0]) * 100
                except:
                    base_prediction = 0
                    base_confidence = 50.0
            else:
                base_prediction = 0
                base_confidence = 50.0
            
            # Extract advanced features
            features = self.feature_extractor.extract_all_features(
                text=article.get('text', ''),
                source=article.get('source'),
                url=article.get('url'),
                title=article.get('title')
            )
            
            # Analyze credibility factors
            credibility_score = self._calculate_credibility_score(features)
            
            # Analyze emotional manipulation
            manipulation_score = self._calculate_manipulation_score(features)
            
            # Combine scores for final prediction
            final_prediction, final_confidence = self._combine_predictions(
                base_prediction, base_confidence, credibility_score, manipulation_score
            )
            
            result = {
                'prediction': 'REAL' if final_prediction == 1 else 'FAKE',
                'confidence': final_confidence,
                'base_prediction': 'REAL' if base_prediction == 1 else 'FAKE',
                'base_confidence': base_confidence,
                'credibility_score': credibility_score,
                'manipulation_score': manipulation_score,
                'features': features,
                'analysis': self._generate_analysis(features, credibility_score, manipulation_score)
            }
            
            results.append(result)
        
        return results
    
    def _calculate_credibility_score(self, features: Dict[str, float]) -> float:
        """Calculate credibility score from features."""
        score = 50.0  # Start neutral
        
        # Source credibility
        if features.get('is_credible_source', 0) > 0:
            score += 30
        elif features.get('is_unreliable_source', 0) > 0:
            score -= 30
        
        # Domain credibility
        if features.get('domain_credibility', 0) > 0:
            score += 20
        elif features.get('domain_unreliability', 0) > 0:
            score -= 20
        
        # Content indicators
        score += features.get('real_indicators_count', 0) * 5
        score -= features.get('fake_indicators_count', 0) * 5
        
        # Citation and attribution
        if features.get('has_attribution', 0) > 0:
            score += 10
        if features.get('has_quotes', 0) > 0:
            score += 5
        
        return max(0, min(100, score))
    
    def _calculate_manipulation_score(self, features: Dict[str, float]) -> float:
        """Calculate emotional manipulation score."""
        score = 0.0
        
        # Emotional language
        score += features.get('emotional_words_count', 0) * 10
        score += features.get('urgency_words_count', 0) * 10
        score += features.get('clickbait_score', 0) * 15
        
        # Excessive punctuation
        score += features.get('exclamation_count', 0) * 2
        score += features.get('uppercase_ratio', 0) * 50
        
        return min(100, score)
    
    def _combine_predictions(self, base_pred: int, base_conf: float, 
                           cred_score: float, manip_score: float) -> Tuple[int, float]:
        """Combine different prediction signals."""
        # Weight the predictions
        text_weight = 0.6
        credibility_weight = 0.3
        manipulation_weight = 0.1
        
        # Convert credibility to fake/real signal
        cred_signal = 1 if cred_score > 60 else 0
        cred_conf = abs(cred_score - 50) * 2  # Convert to confidence
        
        # Convert manipulation to fake signal
        manip_signal = 0 if manip_score > 30 else 1
        manip_conf = min(manip_score * 2, 100)
        
        # Weighted combination
        final_score = (
            (base_pred * base_conf * text_weight) +
            (cred_signal * cred_conf * credibility_weight) +
            (manip_signal * manip_conf * manipulation_weight)
        ) / (base_conf * text_weight + cred_conf * credibility_weight + manip_conf * manipulation_weight)
        
        final_prediction = 1 if final_score > 0.5 else 0
        final_confidence = abs(final_score - 0.5) * 200  # Convert to percentage
        
        return final_prediction, min(100, max(50, final_confidence))
    
    def _generate_analysis(self, features: Dict[str, float], 
                          cred_score: float, manip_score: float) -> List[str]:
        """Generate human-readable analysis."""
        analysis = []
        
        # Credibility analysis
        if cred_score > 80:
            analysis.append("✅ High source credibility")
        elif cred_score < 30:
            analysis.append("⚠️ Low source credibility")
        
        # Content analysis
        if features.get('fake_indicators_count', 0) > 2:
            analysis.append("🚩 Contains multiple fake news indicators")
        
        if features.get('real_indicators_count', 0) > 1:
            analysis.append("📰 Contains journalistic language")
        
        # Emotional manipulation
        if manip_score > 50:
            analysis.append("😡 High emotional manipulation detected")
        elif manip_score > 20:
            analysis.append("⚠️ Some emotional language present")
        
        # Writing style
        if features.get('exclamation_count', 0) > 3:
            analysis.append("❗ Excessive exclamation marks")
        
        if features.get('uppercase_ratio', 0) > 0.1:
            analysis.append("📢 Excessive capitalization")
        
        if not analysis:
            analysis.append("📊 Standard news article characteristics")
        
        return analysis


def install_sentiment_libraries():
    """Install required sentiment analysis libraries."""
    try:
        import subprocess
        import sys
        
        packages = ['textblob', 'vaderSentiment']
        
        for package in packages:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        
        # Download TextBlob corpora
        print("📥 Downloading TextBlob corpora...")
        import textblob
        textblob.download_corpora()
        
        print("✅ All sentiment libraries installed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error installing libraries: {e}")
        return False


def main():
    """
    Main function to demonstrate advanced feature extraction.
    """
    print("🎯 Advanced Feature Extraction for Fake News Detection")
    print("=" * 60)
    
    # Check required libraries
    missing_libs = []
    if not TEXTBLOB_AVAILABLE:
        missing_libs.append('textblob')
    if not VADER_AVAILABLE:
        missing_libs.append('vaderSentiment')
    
    if missing_libs:
        print(f"⚠️ Missing libraries: {missing_libs}")
        install_choice = input("Would you like to install them now? (y/n): ").strip().lower()
        
        if install_choice == 'y':
            if install_sentiment_libraries():
                print("🔄 Please restart the script to use the newly installed libraries.")
                return
            else:
                print("❌ Installation failed. Please install manually:")
                for lib in missing_libs:
                    print(f"   pip install {lib}")
                return
        else:
            print("❌ Cannot proceed without required libraries")
            return
    
    # Create sample articles for testing
    sample_articles = [
        {
            'text': "Breaking news! Scientists have discovered that aliens are living among us and controlling the government through secret mind control technology. This shocking revelation will change everything you know about reality!",
            'title': "SHOCKING: Aliens Control Government Through Mind Control!",
            'source': 'conspiracynews.fake',
            'url': 'http://conspiracynews.fake/aliens-control-gov'
        },
        {
            'text': "According to a new study published in the Journal of Climate Science, researchers at Stanford University have found evidence that renewable energy adoption is accelerating faster than previously predicted. The study, which analyzed data from 50 countries over the past decade, suggests that solar and wind power could account for 60% of global energy production by 2030.",
            'title': "Study: Renewable Energy Adoption Accelerating Globally",
            'source': 'reuters.com',
            'url': 'https://reuters.com/renewable-energy-study'
        }
    ]
    
    # Initialize feature extractor
    extractor = AdvancedFeatureExtractor()
    
    print("🔍 Analyzing sample articles...")
    print("-" * 40)
    
    for i, article in enumerate(sample_articles, 1):
        print(f"\n📰 Article {i}: {article['title']}")
        print(f"Source: {article['source']}")
        
        # Extract features
        features = extractor.extract_all_features(
            text=article['text'],
            source=article['source'],
            url=article['url'],
            title=article['title']
        )
        
        # Display key features
        print(f"\n📊 Key Features:")
        print(f"   Sentiment (TextBlob): {features.get('textblob_polarity', 0):.3f}")
        print(f"   Credible Source: {'Yes' if features.get('is_credible_source', 0) else 'No'}")
        print(f"   Fake Indicators: {features.get('fake_indicators_count', 0)}")
        print(f"   Real Indicators: {features.get('real_indicators_count', 0)}")
        print(f"   Emotional Words: {features.get('emotional_words_count', 0)}")
        print(f"   Clickbait Score: {features.get('clickbait_score', 0)}")
        print(f"   Exclamation Count: {features.get('exclamation_count', 0)}")
    
    # Test enhanced classifier
    print(f"\n🤖 Testing Enhanced Classifier...")
    
    try:
        # Try to load a simple transformer model
        from simple_transformers import SimplifiedTransformerClassifier
        
        model_path = '../models/simple_transformer_all-MiniLM-L6-v2_logistic.joblib'
        if os.path.exists(model_path):
            print("📥 Loading pre-trained transformer model...")
            base_classifier = SimplifiedTransformerClassifier.load(model_path)
            
            enhanced_classifier = EnhancedFakeNewsClassifier(base_classifier)
            results = enhanced_classifier.predict_enhanced(sample_articles)
            
            print(f"\n📈 Enhanced Predictions:")
            for i, (article, result) in enumerate(zip(sample_articles, results), 1):
                print(f"\n{i}. {article['title'][:50]}...")
                print(f"   Final Prediction: {result['prediction']} ({result['confidence']:.1f}%)")
                print(f"   Base Prediction: {result['base_prediction']} ({result['base_confidence']:.1f}%)")
                print(f"   Credibility Score: {result['credibility_score']:.1f}")
                print(f"   Manipulation Score: {result['manipulation_score']:.1f}")
                print(f"   Analysis:")
                for analysis_point in result['analysis']:
                    print(f"     • {analysis_point}")
        else:
            print("⚠️ No pre-trained model found. Train a model first.")
            
    except ImportError:
        print("⚠️ Simple transformer model not available")
    
    print(f"\n✅ Advanced feature extraction demonstration completed!")


if __name__ == "__main__":
    main()