"""
evaluate.py
Evaluation metrics for classification, NER, ABSA, and topic modeling.
"""
import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import List, Dict
import os
import json
from src import setup_logging, LoggerSingleton

def classification_metrics(y_true: List, y_pred: List) -> Dict:
    """Compute precision, recall, F1-score, and confusion matrix."""
    logger = LoggerSingleton.get_logger()
    logger.info("Calculating classification metrics.")
    
    # Handle empty inputs
    if not y_true or not y_pred:
        logger.warning("Empty inputs for classification metrics")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "support": 0
        }
    
    # Get unique classes
    classes = sorted(list(set(y_true + y_pred)))
    
    # Calculate metrics
    metrics = {
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true),
        "support": len(y_true),
        "class_report": classification_report(y_true, y_pred, output_dict=True)
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics["confusion_matrix"] = {
        "matrix": cm.tolist(),
        "classes": classes
    }
    
    return metrics

def evaluate_classification(pred_path: str, gt_path: str, id_col: str, label_col: str, report_path: str):
    """Evaluate classification predictions against ground truth."""
    logger = LoggerSingleton.get_logger()
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Load data
        pred = pd.read_csv(pred_path)
        gt = pd.read_csv(gt_path)
        
        # Check if files are empty
        if pred.empty or gt.empty:
            logger.error(f"Empty prediction or ground truth file: {pred_path}, {gt_path}")
            return
            
        # Check if required columns exist
        if id_col not in pred.columns or id_col not in gt.columns:
            logger.error(f"ID column '{id_col}' not found in prediction or ground truth file")
            return
            
        if f"{label_col}" not in pred.columns:
            logger.error(f"Label column '{label_col}' not found in prediction file")
            return
            
        if f"{label_col}" not in gt.columns:
            logger.error(f"Label column '{label_col}' not found in ground truth file")
            return
        
        # Merge datasets on ID column
        df = pd.merge(pred, gt, on=id_col, suffixes=("_pred", "_true"))
        
        # Check if merge resulted in empty dataframe
        if df.empty:
            logger.error("No matching IDs between prediction and ground truth files")
            return
            
        # Calculate metrics
        metrics = classification_metrics(
            df[f"{label_col}_true"].tolist(), 
            df[f"{label_col}_pred"].tolist()
        )
        
        # Save metrics to CSV for basic metrics
        basic_metrics = {k: v for k, v in metrics.items() 
                        if k not in ["confusion_matrix", "class_report"]}
        pd.DataFrame([basic_metrics]).to_csv(report_path, index=False)
        
        # Save detailed metrics to JSON
        json_path = report_path.replace('.csv', '.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Saved classification metrics to {report_path} and {json_path}")
        
        # Print summary
        print("\nClassification Evaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Support: {metrics['support']}")
        
    except Exception as e:
        logger.error(f"Error in evaluate_classification: {e}")

def evaluate_topic_coherence(cleaned_reviews_path='data/processed/reviews_cleaned.csv', 
                            topic_keywords_path='reports/topics_keywords.csv', 
                            text_col='clean_text', 
                            n_words=10,
                            output_path='reports/topic_coherence.json'):
    """Compute topic coherence using gensim."""
    logger = LoggerSingleton.get_logger()
    logger.info("Calculating topic coherence metrics.")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
        
        # Load cleaned texts
        df = pd.read_csv(cleaned_reviews_path)
        
        if df.empty:
            logger.error(f"Empty reviews file: {cleaned_reviews_path}")
            return {"c_v": 0.0, "u_mass": 0.0, "error": "Empty reviews file"}
            
        if text_col not in df.columns:
            logger.error(f"Text column '{text_col}' not found in reviews file")
            return {"c_v": 0.0, "u_mass": 0.0, "error": f"Missing column: {text_col}"}
        
        # Prepare texts for coherence calculation
        texts = df[text_col].astype(str).apply(lambda x: x.split()).tolist()
        
        # Load topic keywords
        topics_df = pd.read_csv(topic_keywords_path)
        
        if topics_df.empty:
            logger.error(f"Empty topics file: {topic_keywords_path}")
            return {"c_v": 0.0, "u_mass": 0.0, "error": "Empty topics file"}
            
        if 'keywords' not in topics_df.columns:
            logger.error("Keywords column not found in topics file")
            return {"c_v": 0.0, "u_mass": 0.0, "error": "Missing column: keywords"}
        
        # Extract topics
        topics = [row['keywords'].split(', ')[:n_words] for _, row in topics_df.iterrows()]
        
        # Build dictionary
        dictionary = Dictionary(texts)
        
        # Compute coherence metrics
        metrics = {}
        
        # c_v coherence (based on sliding window, semantic similarity)
        try:
            cm_cv = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
            metrics['c_v'] = cm_cv.get_coherence()
        except Exception as e:
            logger.error(f"Error calculating c_v coherence: {e}")
            metrics['c_v'] = 0.0
            metrics['c_v_error'] = str(e)
        
        # u_mass coherence (based on document co-occurrence)
        try:
            cm_umass = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='u_mass')
            metrics['u_mass'] = cm_umass.get_coherence()
        except Exception as e:
            logger.error(f"Error calculating u_mass coherence: {e}")
            metrics['u_mass'] = 0.0
            metrics['u_mass_error'] = str(e)
        
        # Add per-topic coherence
        try:
            metrics['topic_coherence'] = []
            for i, topic in enumerate(topics):
                cm_topic = CoherenceModel(topics=[topic], texts=texts, dictionary=dictionary, coherence='c_v')
                metrics['topic_coherence'].append({
                    'topic_id': i,
                    'coherence': cm_topic.get_coherence(),
                    'keywords': topics_df.iloc[i]['keywords']
                })
        except Exception as e:
            logger.error(f"Error calculating per-topic coherence: {e}")
            metrics['topic_coherence_error'] = str(e)
        
        # Save metrics to JSON
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved topic coherence metrics to {output_path}")
        
        # Print summary
        print("\nTopic Coherence Results:")
        print(f"c_v coherence: {metrics['c_v']:.4f} (higher is better, range typically -1 to 1)")
        print(f"u_mass coherence: {metrics['u_mass']:.4f} (higher is better, usually negative)")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in evaluate_topic_coherence: {e}")
        return {"c_v": 0.0, "u_mass": 0.0, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Evaluation for classification and topic modeling.")
    parser.add_argument('--task', type=str, choices=['classification', 'topic'], required=True, help='Evaluation task')
    parser.add_argument('--pred', type=str, help='Predictions CSV')
    parser.add_argument('--gt', type=str, help='Ground truth CSV (for classification)')
    parser.add_argument('--id_col', type=str, default='review_id', help='ID column')
    parser.add_argument('--label_col', type=str, default='sentiment', help='Label column (for classification)')
    parser.add_argument('--report', type=str, default='../reports/evaluation_report.csv', help='Output report path')
    parser.add_argument('--cleaned_reviews', type=str, default='data/processed/reviews_cleaned.csv', help='Cleaned reviews CSV for topic coherence')
    parser.add_argument('--topic_keywords', type=str, default='reports/topics_keywords.csv', help='Topic keywords CSV for topic coherence')
    args = parser.parse_args()
    setup_logging()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        
        if args.task == 'classification':
            if not args.pred or not args.gt:
                raise ValueError('Both --pred and --gt are required for classification evaluation')
            evaluate_classification(args.pred, args.gt, args.id_col, args.label_col, args.report)
        elif args.task == 'topic':
            evaluate_topic_coherence(args.cleaned_reviews, args.topic_keywords, output_path=args.report.replace('.csv', '.json'))
            
        logger = LoggerSingleton.get_logger()
        logger.info("Evaluation completed.")
    except Exception as e:
        logger = LoggerSingleton.get_logger()
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
