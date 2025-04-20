from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, NGram, 
)
from pyspark.sql.functions import regexp_replace, col, lower, trim

class TextPreprocessor:
    """Handles text cleaning and basic feature extraction"""
    
    def __init__(self, config: dict):
        self.text_col = config.get('text_column', 'text')
        self.stopwords = StopWordsRemover.loadDefaultStopWords("english") + [
            "yelp", "restaurant", "food", "place", "go", "get"
        ]
        
    def create_pipeline(self) -> Pipeline:
        return Pipeline(stages=[
            Tokenizer(inputCol="clean_text", outputCol="words"),
            StopWordsRemover(
                inputCol="words",
                outputCol="filtered_words",
                stopWords=self.stopwords
            ),
            NGram(n=2, inputCol="filtered_words", outputCol="bigrams"),
            NGram(n=3, inputCol="filtered_words", outputCol="trigrams")
        ])

    def clean_text(self, df: DataFrame) -> DataFrame:
        """Basic text cleaning pipeline"""
        return (df
            .withColumn("clean_text", lower(col(self.TEXT_COL)))
            .withColumn("clean_text", regexp_replace(col("clean_text"), r'https?://\S+|www\.\S+', ' '))
            .withColumn("clean_text", regexp_replace(col("clean_text"), r'\S+@\S+', ' '))
            .withColumn("clean_text", regexp_replace(col("clean_text"), r':\)', ' happy '))
            .withColumn("clean_text", regexp_replace(col("clean_text"), r':\(', ' sad '))
            .withColumn("clean_text", regexp_replace(col("clean_text"), r':\/', ' skeptical '))
            .withColumn("clean_text", regexp_replace(col("clean_text"), r"[^a-zA-Z0-9\s]", " "))
            .withColumn("clean_text", regexp_replace(col("clean_text"), r"\b\d+\b", " NUM "))
            .withColumn("clean_text", regexp_replace(col("clean_text"), r"\s+", " "))
            .withColumn("clean_text", trim(col("clean_text")))
        )