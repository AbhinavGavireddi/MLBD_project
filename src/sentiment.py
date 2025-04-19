"""
sentiment.py
Overall sentiment classification using DistilBERT (Hugging Face Transformers).
"""
import argparse
from src import setup_logging, LoggerSingleton
import pandas as pd
from typing import List
from transformers import pipeline
import torch
import os

def classify_sentiment(texts: List[str], model_name: str = "distilbert-base-uncased-finetuned-sst-2-english", batch_size: int = 32, device: int = -1) -> List[str]:
    """Classify sentiment using a fine-tuned DistilBERT model from Hugging Face."""
    logger = LoggerSingleton.get_logger()
    try:
        classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device)
        results = []
        logger.info(f"Processing sentiment for {len(texts)} texts")
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            preds = classifier(batch)
            results.extend([pred['label'].lower() for pred in preds])
        logger.info(f"Finished processing sentiment for {len(texts)} texts")
        return results
    except Exception as e:
        logger.error(f"Error in sentiment classification: {e}")
        logger.warning("Falling back to basic sentiment analysis")
        # Simple fallback: count positive and negative words
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'enjoy', 'nice', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'poor', 'disappointing', 'horrible', 'negative', 'sad']
        
        results = []
        for text in texts:
            text = text.lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            if pos_count > neg_count:
                results.append('positive')
            elif neg_count > pos_count:
                results.append('negative')
            else:
                results.append('neutral')
        return results

def process_reviews(input_csv: str, output_csv: str, text_col: str = "clean_text", id_col: str = "review_id", model_name: str = "distilbert-base-uncased-finetuned-sst-2-english", batch_size: int = 32):
    """Read cleaned reviews, classify sentiment, save output."""
    logger = LoggerSingleton.get_logger()
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} reviews from {input_csv}")
    if id_col not in df.columns:
        df[id_col] = df.index
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    device = 0 if torch.cuda.is_available() else -1
    sentiments = classify_sentiment(df[text_col].astype(str).tolist(), model_name, batch_size, device)
    df_out = df[[id_col]].copy()
    df_out['sentiment'] = sentiments
    df_out.to_csv(output_csv, index=False)
    logger.info(f"Saved sentiment predictions to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Overall sentiment classification using DistilBERT.")
    parser.add_argument('--input', type=str, default='../data/processed/reviews_cleaned.csv', help='Input CSV (cleaned reviews)')
    parser.add_argument('--output', type=str, default='../data/processed/reviews_sentiment.csv', help='Output CSV for sentiment predictions')
    parser.add_argument('--text_col', type=str, default='clean_text', help='Column name for review text')
    parser.add_argument('--id_col', type=str, default='review_id', help='Column name for review id')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased-finetuned-sst-2-english', help='Hugging Face model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()
    setup_logging()
    try:
        process_reviews(args.input, args.output, args.text_col, args.id_col, args.model_name, args.batch_size)
    except Exception as e:
        logger = LoggerSingleton.get_logger()
        logger.error(f"Sentiment classification failed: {e}")

if __name__ == "__main__":
    main()
