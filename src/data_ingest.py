"""
data_ingest.py
Module for loading Yelp Open Dataset (JSON/CSV) into pandas DataFrames.
"""
from src import setup_logging, LoggerSingleton
import pandas as pd
import argparse
import os
import json

def load_reviews(path: str) -> pd.DataFrame:
    """Load Yelp reviews from JSON or CSV."""
    logger = LoggerSingleton.get_logger()
    try:
        if path.endswith('.json'):
            # Use chunking for large JSON files to prevent memory issues
            chunks = []
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if i % 10000 == 0 and i > 0:
                        logger.info(f"Processed {i} lines from {path}")
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON on line {i+1}, skipping")
                        continue
                    # Limit to 100,000 reviews for memory reasons if needed
                    if i >= 100000:
                        logger.warning(f"Limiting to first 100,000 reviews from {path}")
                        break
            df = pd.DataFrame(chunks)
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        logger.info(f"Loaded {len(df)} reviews from {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load reviews from {path}: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['review_id', 'user_id', 'business_id', 'stars', 'text', 'date'])

def load_business(path: str) -> pd.DataFrame:
    """Load Yelp business metadata from JSON or CSV."""
    logger = LoggerSingleton.get_logger()
    try:
        if path.endswith('.json'):
            # Use chunking for large JSON files
            chunks = []
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if i % 10000 == 0 and i > 0:
                        logger.info(f"Processed {i} lines from {path}")
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON on line {i+1}, skipping")
                        continue
                    # Limit to 50,000 businesses for memory reasons if needed
                    if i >= 50000:
                        logger.warning(f"Limiting to first 50,000 businesses from {path}")
                        break
            df = pd.DataFrame(chunks)
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        logger.info(f"Loaded {len(df)} businesses from {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load businesses from {path}: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['business_id', 'name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'categories'])

def save_dataframe(df: pd.DataFrame, out_path: str):
    """Save DataFrame to CSV for reproducibility."""
    logger = LoggerSingleton.get_logger()
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved DataFrame with {len(df)} rows to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {out_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Load Yelp Open Dataset reviews and business data.")
    parser.add_argument('--reviews', type=str, default='../data/raw/yelp_academic_dataset_review.json', help='Path to reviews JSON/CSV')
    parser.add_argument('--business', type=str, default='../data/raw/yelp_academic_dataset_business.json', help='Path to business JSON/CSV')
    parser.add_argument('--out_reviews', type=str, default='../data/processed/reviews.csv', help='Output path for processed reviews CSV')
    parser.add_argument('--out_business', type=str, default='../data/processed/business.csv', help='Output path for processed business CSV')
    args = parser.parse_args()
    setup_logging()
    logger = LoggerSingleton.get_logger()
    
    # Create directories if they don't exist
    for path in [args.out_reviews, args.out_business]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        reviews = load_reviews(args.reviews)
        save_dataframe(reviews, args.out_reviews)
    except Exception as e:
        logger.error(f"Failed to load reviews: {e}")

    try:
        business = load_business(args.business)
        save_dataframe(business, args.out_business)
    except Exception as e:
        logger.error(f"Failed to load business data: {e}")

if __name__ == "__main__":
    main()
