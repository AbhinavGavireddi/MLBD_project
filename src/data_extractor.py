from pyspark.sql import SparkSession, DataFrame
import os
from src import logger


class DataIngestion:
    def __init__(self, config: dict, spark: SparkSession) -> None:
        self.REVIEWS_RAW = config.get("raw_data_path")
        self.REVIEWS = config.get("processed_data_path")
        self.spark = spark
        self.data = self.load_reviews()
        self.save_dataframe()

    def load_reviews(self) -> DataFrame:
        """Load Yelp reviews from JSON or CSV using PySpark."""
        try:
            if self.spark is not None:
                if self.REVIEWS_RAW.endswith(".json"):
                    data = self.spark.read.json(self.REVIEWS_RAW)
                else:
                    raise ValueError(f"Unsupported file format: {self.REVIEWS_RAW}")
                logger.info(
                    f"[PySpark] Loaded {data.count()} reviews from {self.REVIEWS_RAW}"
                )
                return data
        except Exception as e:
            logger.error(f"Failed to load reviews from {self.REVIEWS_RAW}: {e}")
            return None

    def save_dataframe(self):
        """Save DataFrame to parquet for reproducibility using PySpark."""
        try:
            os.makedirs(os.path.dirname(self.REVIEWS), exist_ok=True)
            if hasattr(self.data, "write"):
                self.data.limit(10000).coalesce(1).write.mode("overwrite").parquet(
                    self.REVIEWS, mode="overwrite"
                )
                logger.info(f"[PySpark] Saved DataFrame to {self.REVIEWS}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to {self.REVIEWS}: {e}")
