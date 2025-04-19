"""
Simple orchestrator for the Yelp NLP pipeline.
"""
import os
import sys
from src import (
    data_ingest, preprocess, ner, absa, topic_model, sentiment, evaluate,
    setup_logging, LoggerSingleton, load_config, ensure_dirs
)

class YelpNLPPipeline:
    """
    Modular, easy-to-understand pipeline for Yelp NLP tasks.
    Each step is a clearly named method. Errors are logged and summarized.
    """
    def __init__(self, config_path='config/config.yaml'):
        self.config = load_config(config_path)
        ensure_dirs()
        setup_logging()
        self.logger = LoggerSingleton.get_logger()
        self.steps = []
        self.success = 0
        # Precompute all key paths
        p = self.config['processed_data_path']
        r = self.config['raw_data_path']
        self.paths = {
            'reviews_raw': os.path.join(r, "yelp_academic_dataset_review.json"),
            'business_raw': os.path.join(r, "yelp_academic_dataset_business.json"),
            'reviews': os.path.join(p, "reviews.csv"),
            'business': os.path.join(p, "business.csv"),
            'reviews_cleaned': os.path.join(p, "reviews_cleaned.csv"),
            'entities': os.path.join(p, "reviews_entities.csv"),
            'absa': os.path.join(p, "reviews_absa.csv"),
            'topics': os.path.join(p, "reviews_topics.csv"),
            'topics_keywords': os.path.join(self.config['reports_path'], "topics_keywords.csv"),
            'sentiment': os.path.join(p, "reviews_sentiment.csv")
        }

    def run(self):
        self._step(self.data_ingestion, "Data Ingestion")
        self._step(self.preprocessing, "Preprocessing")
        self._step(self.ner_step, "Named Entity Recognition")
        self._step(self.absa_step, "Aspect-Based Sentiment Analysis")
        self._step(self.topic_modeling, "Topic Modeling")
        self._step(self.sentiment_step, "Sentiment Classification")
        self._step(self.topic_coherence, "Topic Coherence Evaluation")
        self._summary()
        return self.success, len(self.steps)

    def _step(self, func, name):
        try:
            func()
            self.steps.append((name, True))
            self.success += 1
        except Exception as e:
            self.logger.error(f"{name} failed: {e}")
            self.steps.append((name, False))

    def data_ingestion(self):
        if not (os.path.exists(self.paths['reviews_raw']) and os.path.exists(self.paths['business_raw'])):
            raise FileNotFoundError("Missing raw Yelp data files. Please download or check paths.")
        reviews_df = data_ingest.load_reviews(self.paths['reviews_raw'])
        data_ingest.save_dataframe(reviews_df, self.paths['reviews'])
        business_df = data_ingest.load_business(self.paths['business_raw'])
        data_ingest.save_dataframe(business_df, self.paths['business'])

    def preprocessing(self):
        preprocess.process_reviews(self.paths['reviews'], self.paths['reviews_cleaned'])

    def ner_step(self):
        ner.process_reviews(self.paths['reviews_cleaned'], self.paths['entities'])

    def absa_step(self):
        aspects = self.config.get('absa', {}).get('aspects', ["food", "service", "price", "ambience"])
        absa.process_reviews(self.paths['reviews_cleaned'], self.paths['absa'], aspects=aspects)

    def topic_modeling(self):
        lda_conf = self.config.get('lda', {})
        topic_model.run_topic_model(
            input_path=self.paths['reviews_cleaned'],
            output_dist=self.paths['topics'],
            output_keywords=self.paths['topics_keywords'],
            text_col="clean_text",
            id_col="review_id",
            n_topics=lda_conf.get('n_topics', 10),
            max_iter=lda_conf.get('max_iter', 10),
            n_top_words=10,
            method="lda"
        )

    def sentiment_step(self):
        sentiment.process_reviews(self.paths['reviews_cleaned'], self.paths['sentiment'])

    def topic_coherence(self):
        evaluate.evaluate_topic_coherence(self.paths['reviews_cleaned'], self.paths['topics_keywords'])

    def _summary(self):
        print("\nPIPELINE EXECUTION SUMMARY\n" + "="*32)
        for name, ok in self.steps:
            print(f"{name:32} : {'✅' if ok else '❌'}")
        print(f"\nSuccess: {self.success}/{len(self.steps)} steps")

if __name__ == "__main__":
    pipeline = YelpNLPPipeline()
    success_count, total_steps = pipeline.run()
    if success_count == 0:
        sys.exit(1)
