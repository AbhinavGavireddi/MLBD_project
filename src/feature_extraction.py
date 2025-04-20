import re
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.sql.functions import col, size, array, array_intersect, lit, udf
from pyspark.sql.types import FloatType, ArrayType, StringType
from pyspark.sql.functions import greatest, array_contains
from rake_nltk import Rake


class FeatureExtractor:
    """Handles advanced feature extraction"""
    
    def __init__(self, config: dict):
        self.config = config
        self.cv_model = None
        
    def create_tfidf_pipeline(self) -> Pipeline:
        return Pipeline(stages=[
            CountVectorizer(
                inputCol="filtered_words",
                outputCol="tf",
                minDF=5,
                vocabSize=10000
            ),
            IDF(inputCol="tf", outputCol="tfidf")
        ])
        
    def add_sentiment_features(self, df: DataFrame) -> DataFrame:
        """Calculate sentiment metrics"""
        pos_words = ['good', 'great', 'excellent', 'amazing', 'awesome']
        neg_words = ['bad', 'worst', 'terrible', 'awful', 'horrible']
        
        return (
            df
            .withColumn("positive_count", size(array_intersect(col("filtered_words"), array([lit(w) for w in pos_words]))))
            .withColumn("negative_count", size(array_intersect(col("filtered_words"), array([lit(w) for w in neg_words]))))
            .withColumn("sentiment_score", col("positive_count") / (col("negative_count") + 1))
            )
    def add_readability_scores(self, df: DataFrame) -> DataFrame:
        """Calculate Flesch-Kincaid readability scores"""
        @udf(FloatType())
        def flesch_kincaid_udf(text):
            sentences = len(re.split(r'[.!?]+', text))
            words = len(text.split())
            syllables = sum([max(1, sum(1 for char in word if char in 'aeiouy')) for word in text.split()])
            return 206.835 - 1.015*(words/max(sentences,1)) - 84.6*(syllables/max(words,1))
        
        return df.withColumn("readability_score", flesch_kincaid_udf(col("clean_text")))
    
    def detect_emotions(self, df: DataFrame) -> DataFrame:
        """Detect emotional tones in reviews"""
        emotion_map = {
            "anger": ["angry", "mad", "furious"],
            "joy": ["happy", "joyful", "delighted"],
            "surprise": ["surprised", "shocked", "amazed"]
        }
        
        for emotion, keywords in emotion_map.items():
            df = df.withColumn(
                f"emotion_{emotion}",
                greatest(*[array_contains(col("filtered_words"), k) for k in keywords])
            )
        
        return df
    
    def extract_key_phrases(self, df: DataFrame) -> DataFrame:
        """Extract key phrases using RAKE algorithm"""
        @udf(ArrayType(StringType()))
        def rake_phrases(text):
            r = Rake()
            r.extract_keywords_from_text(text)
            return r.get_ranked_phrases()[:3]
        
        return df.withColumn("key_phrases", rake_phrases(col("clean_text")))