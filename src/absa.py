"""
absa.py
Aspect-Based Sentiment Analysis using PyABSA.
"""
import argparse
import pandas as pd
from typing import List, Dict
from pyabsa import available_checkpoints, ATEPCCheckpointManager
import os
import re
from src import setup_logging, LoggerSingleton

def analyze_aspects(texts: List[str], aspects: List[str], model_path: str = None, batch_size: int = 32) -> List[Dict]:
    """Analyze aspect-based sentiment for each review using PyABSA."""
    logger = LoggerSingleton.get_logger()
    try:
        # Download or use provided model
        if model_path is None:
            checkpoints = available_checkpoints()
            # Use the first English checkpoint (usually 'english')
            model_path = [c for c in checkpoints if 'english' in c][0]
        aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(model_path)
        results = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            try:
                preds = aspect_extractor.extract_aspect(inference_source=batch, print_result=False, pred_sentiment=True)
                for pred in preds:
                    aspect_sent_dict = {aspect: 'none' for aspect in aspects}
                    if 'aspect' in pred and 'sentiment' in pred:
                        for aspect_term, sentiment in zip(pred['aspect'], pred['sentiment']):
                            # Improved aspect matching with word boundaries
                            matched = False
                            for predefined in aspects:
                                # Check for exact match or word boundary match
                                if (predefined.lower() == aspect_term.lower() or 
                                    re.search(r'\b' + re.escape(predefined.lower()) + r'\b', aspect_term.lower())):
                                    aspect_sent_dict[predefined] = sentiment
                                    matched = True
                                    break
                            
                            # If no direct match, try partial match
                            if not matched:
                                for predefined in aspects:
                                    if predefined.lower() in aspect_term.lower():
                                        aspect_sent_dict[predefined] = sentiment
                                        break
                    results.append(aspect_sent_dict)
            except Exception as batch_error:
                logger.error(f"Error processing batch {i//batch_size + 1}: {batch_error}")
                # Add default values for failed batch
                for _ in batch:
                    results.append({aspect: 'none' for aspect in aspects})
        
        return results
    except Exception as e:
        logger.error(f"ABSA analysis failed: {e}")
        # Return default values on failure
        return [{aspect: 'none' for aspect in aspects} for _ in texts]

def process_reviews(input_csv: str, output_csv: str, aspects: List[str], text_col: str = "clean_text", id_col: str = "review_id", model_path: str = None, batch_size: int = 32):
    """Read cleaned reviews, analyze aspect sentiment, save output."""
    logger = LoggerSingleton.get_logger()
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} reviews from {input_csv}")
        if id_col not in df.columns:
            df[id_col] = df.index
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        aspect_sentiments = analyze_aspects(df[text_col].astype(str).tolist(), aspects, model_path, batch_size)
        df_out = df[[id_col]].copy()
        for aspect in aspects:
            df_out[aspect] = [a[aspect] for a in aspect_sentiments]
        df_out.to_csv(output_csv, index=False)
        logger.info(f"Saved aspect sentiments to {output_csv}")
    except Exception as e:
        logger.error(f"Failed to process reviews for ABSA: {e}")
        # Create minimal output with just IDs to prevent pipeline failure
        try:
            if 'df' in locals() and id_col in df.columns:
                minimal_df = df[[id_col]].copy()
                for aspect in aspects:
                    minimal_df[aspect] = 'none'
                minimal_df.to_csv(output_csv, index=False)
                logger.warning(f"Saved fallback aspect sentiments to {output_csv}")
        except Exception as fallback_error:
            logger.error(f"Even fallback save failed: {fallback_error}")

def main():
    parser = argparse.ArgumentParser(description="Aspect-based sentiment analysis using PyABSA.")
    parser.add_argument('--input', type=str, default='../data/processed/reviews_cleaned.csv', help='Input CSV (cleaned reviews)')
    parser.add_argument('--output', type=str, default='../data/processed/reviews_absa.csv', help='Output CSV for aspect sentiments')
    parser.add_argument('--aspects', nargs='+', default=['food', 'service', 'price', 'ambience'], help='List of aspects')
    parser.add_argument('--text_col', type=str, default='clean_text', help='Column name for review text')
    parser.add_argument('--id_col', type=str, default='review_id', help='Column name for review id')
    parser.add_argument('--model_path', type=str, default=None, help='Path to PyABSA checkpoint (optional)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()
    setup_logging()
    try:
        process_reviews(args.input, args.output, args.aspects, args.text_col, args.id_col, args.model_path, args.batch_size)
    except Exception as e:
        logger = LoggerSingleton.get_logger()
        logger.error(f"ABSA failed: {e}")

if __name__ == "__main__":
    main()
