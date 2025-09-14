"""
Dataset downloader and preprocessor for fake news detection.
This module handles downloading datasets from Kaggle and other sources.
"""

import os
import pandas as pd
import requests
import zipfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DatasetDownloader:
    """
    A class to download and prepare fake news datasets.
    """
    
    def __init__(self, data_dir='../data'):
        """
        Initialize the dataset downloader.
        
        Args:
            data_dir (str): Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Available datasets (URLs for direct download)
        self.datasets = {
            'fake_news_net': {
                'url': 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/Data/PolitiFact/Real/gossipcop_real.csv',
                'description': 'FakeNewsNet dataset - PolitiFact real news',
                'format': 'csv'
            },
            'isot_fake_news': {
                'url': 'https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/main/fake_or_real_news.csv',
                'description': 'ISOT Fake News Dataset',
                'format': 'csv'
            },
            'sample_combined': {
                'description': 'Combined sample dataset from multiple sources',
                'format': 'generated'
            }
        }
    
    def download_dataset(self, dataset_name='sample_combined'):
        """
        Download a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset to download
            
        Returns:
            str: Path to the downloaded dataset file
        """
        if dataset_name not in self.datasets:
            print(f"❌ Dataset '{dataset_name}' not available.")
            print(f"Available datasets: {list(self.datasets.keys())}")
            return None
        
        dataset_info = self.datasets[dataset_name]
        
        if dataset_name == 'sample_combined':
            return self._create_enhanced_sample_dataset()
        
        print(f"📥 Downloading {dataset_info['description']}...")
        
        try:
            # Download the dataset
            response = requests.get(dataset_info['url'], timeout=30)
            response.raise_for_status()
            
            # Save to file
            filename = f"{dataset_name}.{dataset_info['format']}"
            filepath = self.data_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Dataset downloaded to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("📝 Creating enhanced sample dataset instead...")
            return self._create_enhanced_sample_dataset()
    
    def _create_enhanced_sample_dataset(self):
        """
        Create an enhanced sample dataset with more examples.
        """
        print("🔧 Creating enhanced sample dataset...")
        
        # Expanded fake news examples (more diverse)
        fake_news = [
            # Conspiracy theories
            "Scientists have discovered that aliens are living among us and controlling the government through secret mind control",
            "Secret government documents reveal that the moon landing was filmed in Hollywood studios by Stanley Kubrick",
            "New research shows vaccines contain microchips designed for population control and surveillance by world governments",
            "Breaking: Time travel machine invented by teenager in garage allows communication with future civilizations",
            "Hidden cure for cancer discovered 50 years ago but suppressed by pharmaceutical companies to maximize profits",
            
            # Health misinformation
            "Miracle weight loss pill discovered that allows you to lose 50 pounds in one week without diet or exercise",
            "Doctors hate this one weird trick that cures all diseases including diabetes, cancer, and heart disease",
            "Eating chocolate for breakfast proven to cure depression and increase IQ by 50 points according to secret study",
            "Drinking bleach shown to cure coronavirus and all viral infections in underground medical trials",
            "New superfood discovered in Amazon rainforest grants immortality and reverses aging process completely",
            
            # Technology misinformation
            "5G towers confirmed to cause coronavirus and are being used for mass mind control by tech companies",
            "Artificial intelligence has become sentient and is secretly controlling all social media to manipulate elections",
            "Internet will shut down permanently next week due to solar storm predicted by ancient Mayan calendar",
            "Smartphones proven to steal thoughts and transmit them to government surveillance agencies worldwide",
            "Virtual reality headsets discovered to trap users' souls in digital dimension according to leaked documents",
            
            # Political misinformation
            "Secret society of billionaires controls all world governments and decides election outcomes in advance",
            "President admits on hidden recording that democracy is fake and voters have no real power",
            "Foreign countries using weather control technology to create natural disasters and influence politics",
            "Underground tunnels discovered beneath major cities used by elite politicians for illegal activities",
            "Voting machines proven to be hacked by teenagers changing election results across multiple countries",
            
            # Economic misinformation
            "Global economic collapse planned for next month by international banking conspiracy to reset world currency",
            "Bitcoin secretly controlled by single individual who can crash entire cryptocurrency market at will",
            "Major corporations using subliminal advertising to force consumers to buy products against their will",
            "Stock market is completely fake and all prices are predetermined by computer algorithms",
            "Gold reserves in major countries discovered to be fake painted rocks according to whistleblower",
            
            # Science misinformation
            "Climate change proven to be hoax created by scientists to secure more research funding",
            "Flat Earth theory confirmed by NASA insider who reveals space agency has been lying for decades",
            "Evolution theory debunked by new fossil evidence showing humans coexisted with dinosaurs",
            "Gravity is not real and objects fall due to electromagnetic forces controlled by secret technology",
            "Dinosaurs never existed and all fossils were planted by government to support false evolutionary timeline"
        ]
        
        # Expanded real news examples (more diverse)
        real_news = [
            # Local government and politics
            "City council approves new budget allocation for public transportation improvements and infrastructure development",
            "Local mayor announces plans to expand affordable housing program to help address homelessness crisis",
            "State legislature passes bill increasing funding for public education and teacher salary improvements",
            "Municipal water department completes upgrades to treatment facility ensuring safe drinking water for residents",
            "County commissioners approve new recycling program to reduce waste and promote environmental sustainability",
            
            # Business and economy
            "Technology company announces plans to expand operations and hire 500 new employees in the region",
            "Stock market closes up 3% following positive quarterly earnings reports from major technology sector",
            "Local small business receives grant to develop innovative renewable energy solutions for rural communities",
            "Manufacturing plant implements new safety protocols reducing workplace accidents by 40% this year",
            "Regional bank offers new low-interest loan program to support first-time homebuyers in the area",
            
            # Health and medicine
            "Medical researchers at university publish study on effectiveness of new diabetes treatment approach",
            "Health officials recommend annual flu vaccination for all adults over 18 years old this season",
            "Hospital announces successful completion of clinical trial for innovative cancer therapy treatment",
            "Public health department launches campaign to promote mental health awareness and support services",
            "Medical school receives federal grant to train more healthcare professionals for underserved communities",
            
            # Education and research
            "University researchers develop new method for detecting early signs of Alzheimer's disease",
            "Local school district implements new STEM education program to prepare students for technology careers",
            "College announces scholarship program for students from low-income families pursuing science degrees",
            "Education department launches initiative to improve literacy rates in rural areas through mobile libraries",
            "Research team publishes findings on sustainable agriculture techniques for climate change adaptation",
            
            # Environment and science
            "Environmental group partners with local businesses to reduce plastic waste through recycling initiatives",
            "Scientists discover new species of butterfly in protected national forest reserve",
            "Renewable energy project begins construction of solar farm expected to power 10000 homes",
            "Climate research station reports data showing gradual recovery of endangered species population",
            "National park service implements new conservation program to protect native wildlife habitats",
            
            # Technology and innovation
            "Software company develops new app to help elderly residents access healthcare services remotely",
            "Engineering students create device that converts ocean plastic waste into usable building materials",
            "Internet infrastructure improvements bring high-speed broadband to rural communities for first time",
            "Researchers develop more efficient battery technology for electric vehicles and renewable energy storage",
            "Tech startup launches platform connecting local farmers directly with consumers to reduce food waste",
            
            # Transportation and infrastructure
            "Transportation authority announces schedule improvements for public bus routes serving suburban areas",
            "Construction begins on new bridge designed to reduce traffic congestion and improve safety",
            "Airport authority completes runway expansion project to accommodate increased passenger traffic",
            "Department of transportation awards contract for highway maintenance and pothole repair program",
            "Railway company introduces new electric trains to reduce emissions and improve passenger comfort"
        ]
        
        # Create DataFrame
        data = []
        
        # Add fake news samples
        for text in fake_news:
            data.append({
                'text': text,
                'label': 0,  # 0 for fake
                'title': text[:50] + "...",
                'subject': 'fake',
                'date': '2025-09-15'
            })
        
        # Add real news samples
        for text in real_news:
            data.append({
                'text': text,
                'label': 1,  # 1 for real
                'title': text[:50] + "...",
                'subject': 'real',
                'date': '2025-09-15'
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        filepath = self.data_dir / 'enhanced_sample_dataset.csv'
        df.to_csv(filepath, index=False)
        
        print(f"✅ Enhanced sample dataset created with {len(df)} articles")
        print(f"   - Fake news: {len(fake_news)} articles")
        print(f"   - Real news: {len(real_news)} articles")
        print(f"   - Saved to: {filepath}")
        
        return str(filepath)
    
    def load_dataset(self, filepath):
        """
        Load and validate a dataset.
        
        Args:
            filepath (str): Path to the dataset file
            
        Returns:
            pd.DataFrame: Loaded and validated dataset
        """
        try:
            df = pd.read_csv(filepath)
            
            # Standardize column names
            column_mapping = {
                'title': 'text',
                'content': 'text',
                'article': 'text',
                'news': 'text',
                'fake': 'label',
                'target': 'label',
                'class': 'label'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Ensure required columns exist
            if 'text' not in df.columns:
                raise ValueError("Dataset must have a 'text' column")
            
            if 'label' not in df.columns:
                # Try to infer labels from other columns
                if 'subject' in df.columns:
                    df['label'] = df['subject'].map({'fake': 0, 'real': 1}).fillna(0)
                else:
                    raise ValueError("Dataset must have a 'label' column or 'subject' column")
            
            # Clean labels (ensure 0/1 format)
            if df['label'].dtype == 'object':
                label_mapping = {
                    'fake': 0, 'FAKE': 0, 'false': 0, 'FALSE': 0,
                    'real': 1, 'REAL': 1, 'true': 1, 'TRUE': 1
                }
                df['label'] = df['label'].map(label_mapping).fillna(0)
            
            # Remove rows with missing values
            df = df.dropna(subset=['text', 'label'])
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['text'])
            
            print(f"📊 Dataset loaded successfully:")
            print(f"   - Total articles: {len(df)}")
            print(f"   - Fake articles: {len(df[df['label'] == 0])}")
            print(f"   - Real articles: {len(df[df['label'] == 1])}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def download_kaggle_dataset(self, dataset_name, username=None, key=None):
        """
        Download dataset from Kaggle (requires Kaggle API setup).
        
        Args:
            dataset_name (str): Kaggle dataset name (e.g., 'clmentbisaillon/fake-and-real-news-dataset')
            username (str): Kaggle username
            key (str): Kaggle API key
        """
        try:
            # Try to import kaggle
            import kaggle
            
            print(f"📥 Downloading from Kaggle: {dataset_name}")
            
            # Download dataset
            download_path = self.data_dir / 'kaggle_data'
            download_path.mkdir(exist_ok=True)
            
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(download_path), 
                unzip=True
            )
            
            print(f"✅ Kaggle dataset downloaded to: {download_path}")
            
            # Find CSV files
            csv_files = list(download_path.glob('*.csv'))
            if csv_files:
                print(f"📁 Found CSV files: {[f.name for f in csv_files]}")
                return str(csv_files[0])  # Return first CSV file
            else:
                print("❌ No CSV files found in downloaded dataset")
                return None
                
        except ImportError:
            print("❌ Kaggle library not installed. Install with: pip install kaggle")
            print("📝 Using enhanced sample dataset instead...")
            return self._create_enhanced_sample_dataset()
        except Exception as e:
            print(f"❌ Error downloading from Kaggle: {e}")
            print("📝 Make sure you have Kaggle API credentials set up")
            print("📝 Visit: https://www.kaggle.com/docs/api")
            print("📝 Using enhanced sample dataset instead...")
            return self._create_enhanced_sample_dataset()


def main():
    """
    Main function to demonstrate dataset downloading.
    """
    print("📥 Fake News Dataset Downloader")
    print("=" * 50)
    
    downloader = DatasetDownloader()
    
    print("\n🎯 Available options:")
    print("1. Enhanced sample dataset (recommended for testing)")
    print("2. Download from Kaggle (requires API setup)")
    print("3. Download from public URLs")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        filepath = downloader.download_dataset('sample_combined')
    elif choice == '2':
        dataset_name = input("Enter Kaggle dataset name (e.g., 'clmentbisaillon/fake-and-real-news-dataset'): ").strip()
        if dataset_name:
            filepath = downloader.download_kaggle_dataset(dataset_name)
        else:
            print("❌ No dataset name provided")
            filepath = downloader.download_dataset('sample_combined')
    elif choice == '3':
        print("🔄 Attempting to download from public URLs...")
        filepath = downloader.download_dataset('isot_fake_news')
    else:
        print("❌ Invalid choice. Using enhanced sample dataset...")
        filepath = downloader.download_dataset('sample_combined')
    
    if filepath:
        print(f"\n✅ Dataset ready at: {filepath}")
        
        # Load and display sample
        df = downloader.load_dataset(filepath)
        if df is not None:
            print(f"\n📋 Sample articles:")
            print("-" * 50)
            for i, row in df.head(3).iterrows():
                label = "FAKE" if row['label'] == 0 else "REAL"
                print(f"{i+1}. [{label}] {row['text'][:100]}...")
            
            return filepath
    else:
        print("❌ Failed to download dataset")
        return None


if __name__ == "__main__":
    main()