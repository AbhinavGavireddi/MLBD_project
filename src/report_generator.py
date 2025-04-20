from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import os
import logging

logger = logging.getLogger(__name__)

class InsightReporter:
    """Generates actionable business reports"""
    
    def __init__(self, config: dict):
        self.output_dir = config.get('output_dir', './business_insights')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_all_reports(self, metrics_df: DataFrame) -> None:
        """Generate comprehensive business reports"""
        self._save_report(
            metrics_df.select(
                "business_id", "yearly_avg_rating", "yearly_review_count",
                "food_quality_rate", "service_rate"
            ), 
            "core_metrics"
        )
        self._save_report(
            metrics_df.select(
                "business_id", "unique_customers",
                "avg_customer_engagement", "competitor_rank"
            ),
            "customer_analysis"
        )
        self._save_report(
            metrics_df.selectExpr(
                "business_id",
                "posexplode(key_phrases) as (phrase_rank, key_phrase)"
            ).filter(col("phrase_rank") < 3),
            "top_key_phrases"
        )
    
    def _save_report(self, df: DataFrame, name: str) -> None:
        """Save report with proper formatting"""
        path = os.path.join(self.output_dir, f"{name}.parquet")
        df.write.mode("overwrite").parquet(path)
        logger.info(f"Saved {name} report with {df.count()} records")