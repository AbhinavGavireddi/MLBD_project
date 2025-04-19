"""
topic_model.py
Topic modeling using LDA/NMF in scikit-learn.
"""
import argparse
from src import setup_logging, LoggerSingleton
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def run_topic_model(input_path: str, output_dist: str, output_keywords: str, text_col: str = "clean_text", id_col: str = "review_id", n_topics: int = 10, max_iter: int = 10, n_top_words: int = 10, method: str = "lda"):
    logger = LoggerSingleton.get_logger()
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} reviews from {input_path}")
    texts = df[text_col].astype(str).tolist()
    ids = df[id_col] if id_col in df.columns else df.index
    if method == "lda":
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)
        model = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, random_state=42)
    else:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        model = NMF(n_components=n_topics, max_iter=max_iter, random_state=42)
    logger.info(f"Running {method.upper()} topic modeling with {n_topics} topics")
    topic_dist = model.fit_transform(X)
    topic_df = pd.DataFrame(topic_dist, columns=[f"topic_{i}" for i in range(n_topics)])
    topic_df[id_col] = ids
    topic_df.to_csv(output_dist, index=False)
    logger.info(f"Saved topic distributions to {output_dist}")
    # Save keywords
    words = vectorizer.get_feature_names_out()
    keywords = []
    for i, topic in enumerate(model.components_):
        top = [words[j] for j in topic.argsort()[-n_top_words:][::-1]]
        keywords.append({"topic": i, "keywords": ", ".join(top)})
    pd.DataFrame(keywords).to_csv(output_keywords, index=False)
    logger.info(f"Saved topic keywords to {output_keywords}")

def main():
    parser = argparse.ArgumentParser(description="Topic modeling on Yelp reviews using LDA or NMF.")
    parser.add_argument('--input', type=str, default='../data/processed/reviews_cleaned.csv', help='Input CSV (cleaned reviews)')
    parser.add_argument('--output_dist', type=str, default='../data/processed/reviews_topics.csv', help='Output CSV for topic distributions')
    parser.add_argument('--output_keywords', type=str, default='../reports/topics_keywords.csv', help='Output CSV for topic keywords')
    parser.add_argument('--text_col', type=str, default='clean_text', help='Column name for review text')
    parser.add_argument('--id_col', type=str, default='review_id', help='Column name for review id')
    parser.add_argument('--n_topics', type=int, default=10, help='Number of topics')
    parser.add_argument('--max_iter', type=int, default=10, help='Max iterations')
    parser.add_argument('--n_top_words', type=int, default=10, help='Number of top words per topic')
    parser.add_argument('--method', type=str, choices=['lda', 'nmf'], default='lda', help='Topic modeling method')
    args = parser.parse_args()
    setup_logging()
    try:
        run_topic_model(args.input, args.output_dist, args.output_keywords, args.text_col, args.id_col, args.n_topics, args.max_iter, args.n_top_words, args.method)
    except Exception as e:
        logger = LoggerSingleton.get_logger()
        logger.error(f"Topic modeling failed: {e}")

if __name__ == "__main__":
    main()
