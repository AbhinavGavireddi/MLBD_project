"""
preprocess.py
Text cleaning and preprocessing using spaCy, NLTK, etc.
"""
import spacy
import re
import argparse
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src import setup_logging, LoggerSingleton
import pandas as pd
import os

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import sys
    print("Error: spaCy model 'en_core_web_sm' not found. Please install it with:")
    print("python -m spacy download en_core_web_sm")
    sys.exit(1)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Lowercase, remove special chars, handle emojis, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r"[\W_]+", " ", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def preprocess_text(text: str) -> List[str]:
    """Tokenize, remove stopwords, lemmatize."""
    if not text:
        return []
    
    try:
        doc = nlp(text)
        tokens = [lemmatizer.lemmatize(token.text) for token in doc 
                 if token.text not in stop_words and token.is_alpha and len(token.text) > 1]
        return tokens
    except Exception as e:
        logger = LoggerSingleton.get_logger()
        logger.error(f"Error in text preprocessing: {e}")
        # Fallback to simple tokenization
        return [lemmatizer.lemmatize(word) for word in text.split() 
                if word not in stop_words and word.isalpha() and len(word) > 1]

def process_reviews(input_csv: str, output_csv: str, text_col: str = "text", id_col: str = "review_id"):
    logger = LoggerSingleton.get_logger()
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} reviews from {input_csv}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        if id_col not in df.columns:
            df[id_col] = df.index
            logger.info(f"Created {id_col} column as it was missing")
            
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in the input CSV")
            
        # Handle missing values
        df[text_col] = df[text_col].fillna("")
        
        logger.info("Cleaning text...")
        df['clean_text'] = df[text_col].astype(str).apply(clean_text)
        
        logger.info("Tokenizing and lemmatizing...")
        df['tokens'] = df['clean_text'].apply(preprocess_text)
        
        # Filter by review length (normalize) but don't be too strict
        original_count = len(df)
        df = df[df['tokens'].apply(lambda x: len(x) >= 3)]
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} reviews with fewer than 3 tokens")
        
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(df)} cleaned reviews to {output_csv}")
    except Exception as e:
        logger.error(f"Error in process_reviews: {e}")
        # Create minimal output to prevent pipeline failure
        try:
            if 'df' in locals() and id_col in df.columns:
                minimal_df = df[[id_col]].copy()
                minimal_df['clean_text'] = ""
                minimal_df['tokens'] = minimal_df['clean_text'].apply(lambda x: [])
                minimal_df.to_csv(output_csv, index=False)
                logger.warning(f"Saved fallback cleaned reviews to {output_csv}")
        except Exception as fallback_error:
            logger.error(f"Even fallback save failed: {fallback_error}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Yelp reviews: clean, tokenize, lemmatize.")
    parser.add_argument('--input', type=str, default='../data/processed/reviews.csv', help='Input CSV (from data_ingest)')
    parser.add_argument('--output', type=str, default='../data/processed/reviews_cleaned.csv', help='Output CSV for cleaned reviews')
    parser.add_argument('--text_col', type=str, default='text', help='Column name for review text')
    args = parser.parse_args()
    setup_logging()
    try:
        process_reviews(args.input, args.output, args.text_col)
    except Exception as e:
        logger = LoggerSingleton.get_logger()
        logger.error(f"Preprocessing failed: {e}")

if __name__ == "__main__":
    main()
