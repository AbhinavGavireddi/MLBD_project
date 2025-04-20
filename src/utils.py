from src import logger
from pyspark.sql import SparkSession, DataFrame
import yaml

def load_json(path: str, spark: SparkSession = None) -> DataFrame:
    """Load Yelp reviews from JSON or CSV using PySpark if available."""
    try:
        if spark is not None:
            if path.endswith('.json'):
                df = spark.read.json(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
            logger.info(f"[PySpark] Loaded {df.count()} reviews from {path}")
            return df
    except Exception as e:
        logger.error(f"Failed to load reviews from {path}: {e}")
        return None


def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        return None
    

def get_spark_session() -> SparkSession:
    """Get a Spark session with default configuration."""
    return (
        SparkSession.builder
        .appName("YelpBusinessIntelligence")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
        )