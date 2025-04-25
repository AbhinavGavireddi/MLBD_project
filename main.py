from pyspark.sql import SparkSession
from src import logger
from src.utils import load_config, get_spark_session
from src.data_extractor import DataIngestion
from src.preprocessor import TextPreprocessor
from src.business_metrics_calculator import BusinessAnalytics
from src.report_generator import InsightReporter
from src.feature_extraction import FeatureExtractor


class YelpBusinessIntelligence:
    """End-to-end business intelligence pipeline"""
    
    def __init__(self, config: dict, spark: SparkSession):
        self.spark = spark
        self.config = config
        
        # Initialize components
        self.ingestion = DataIngestion(config, spark)
        self.preprocessor = TextPreprocessor(config)
        self.feature_engineer = FeatureExtractor(config)
        self.analytics = BusinessAnalytics(config)
        self.reporter = InsightReporter(config)
        
    def execute(self) -> None:
        """Execute complete pipeline"""
        try:
            # 1. Data Preparation
            clean_df = self.preprocessor.clean_text(self.ingestion.data)
            clean_df.cache()

            # 2. Text Processing
            text_pipeline = self.preprocessor.create_pipeline()
            processed_df = text_pipeline.fit(clean_df).transform(clean_df)
            processed_df.cache()
            clean_df.unpersist()
            
            # 3. Feature Engineering
            featured_df = self.feature_engineer.create_tfidf_pipeline().fit(processed_df).transform(processed_df)
            featured_df = self.feature_engineer.add_sentiment_features(featured_df)
            featured_df = self.feature_engineer.add_readability_scores(featured_df)
            featured_df = self.feature_engineer.detect_emotions(featured_df)
            featured_df = self.feature_engineer.extract_key_phrases(featured_df)
            featured_df.cache()
            processed_df.unpersist()
            
            # 4. Business Analytics
            metrics_df = self.analytics.calculate_core_metrics(featured_df)
            metrics_df.cache()
            featured_df.unpersist()
            
            # 5. Reporting
            self.reporter.generate_all_reports(metrics_df)
            
            logger.info("Business intelligence pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

pipeline = YelpBusinessIntelligence(load_config('config/config.yaml'), get_spark_session())
pipeline.execute()