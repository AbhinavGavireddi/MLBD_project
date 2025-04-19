"""
ner.py
Named Entity Recognition using spaCy.
"""
import spacy
import argparse
from typing import List, Dict
from src import setup_logging, LoggerSingleton
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> List[Dict]:
    """Extract named entities from text."""
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def process_reviews(input_csv: str, output_csv: str, text_col: str = "clean_text", id_col: str = "review_id"):
    """Read cleaned reviews, extract entities, save output."""
    df = pd.read_csv(input_csv)
    logger = LoggerSingleton.get_logger()
    logger.info(f"Loaded {len(df)} reviews from {input_csv}")
    if id_col not in df.columns:
        df[id_col] = df.index  # fallback if no review_id column
    
    logger.info(f"Starting entity extraction for {len(df)} reviews")
    entities = [extract_entities(text) for text in df[text_col].astype(str)]
    logger.info(f"Completed entity extraction. Found entities in {sum(1 for e in entities if e)} reviews")
    
    df_out = df[[id_col]].copy()
    df_out["entities"] = entities
    df_out.to_csv(output_csv, index=False)
    logger.info(f"Saved NER results to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Extract named entities from Yelp reviews using spaCy.")
    parser.add_argument('--input', type=str, default='../data/processed/reviews_cleaned.csv', help='Input CSV (cleaned reviews)')
    parser.add_argument('--output', type=str, default='../data/processed/reviews_entities.csv', help='Output CSV for entities')
    parser.add_argument('--text_col', type=str, default='clean_text', help='Column name for review text')
    parser.add_argument('--id_col', type=str, default='review_id', help='Column name for review id')
    args = parser.parse_args()
    setup_logging()
    try:
        process_reviews(args.input, args.output, args.text_col, args.id_col)
    except Exception as e:
        logger = LoggerSingleton.get_logger()
        logger.error(f"NER failed: {e}")

if __name__ == "__main__":
    main()
