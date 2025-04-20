from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, year, month, avg, expr, array_contains, greatest, count,
    rank, countDistinct
)
from pyspark.sql.window import Window

class BusinessAnalytics:
    """Calculates core business performance metrics"""
    
    def __init__(self, config: dict):
        self.date_col = config.get('date_column', 'date')
        self.business_col = config.get('business_id_column', 'business_id')
        self.rating_col = config.get('rating_column', 'stars')
        self.user_col = config.get('user_id_column', 'user_id')
        
    def calculate_core_metrics(self, df: DataFrame) -> DataFrame:
        """Calculate all business-critical metrics"""
        df = self._add_temporal_metrics(df)
        df = self._add_aspect_analysis(df)
        df = self._add_customer_behavior(df)
        return self._add_competitive_insights(df)
    
    def _add_temporal_metrics(self, df: DataFrame) -> DataFrame:
        """Analyze trends over time"""
        window = Window.partitionBy(self.business_col).orderBy(self.date_col)
        return (
            df
            .withColumn("rolling_avg_rating", avg(col(self.rating_col)).over(window.rowsBetween(-6, 0)))
            .groupBy(self.business_col, year(self.date_col).alias("year"))
            .agg(
                avg(self.rating_col).alias("yearly_avg_rating"),
                count("*").alias("yearly_review_count"),
                expr("percentile_approx(stars, 0.5)").alias("median_rating")
            )
        )
    
    def _add_aspect_analysis(self, df: DataFrame) -> DataFrame:
        """Analyze mentions of key business aspects"""
        aspects = {
            "food_quality": ["fresh", "tasty", "delicious", "flavor"],
            "service": ["friendly", "staff", "service", "waitress"],
            "ambiance": ["atmosphere", "decor", "music", "lighting"]
        }
        
        for aspect, terms in aspects.items():
            df = df.withColumn(
                f"{aspect}_mentions",
                greatest(*[array_contains(col("bigrams"), t) for t in terms])
            )
        
        return df.groupBy(self.business_col).agg(
            *[avg(f"{a}_mentions").alias(f"{a}_rate") for a in aspects.keys()]
        )
    
    def _add_customer_behavior(self, df: DataFrame) -> DataFrame:
        """Analyze customer engagement patterns"""
        return df.groupBy(self.user_col).agg(
            count("*").alias("total_reviews"),
            avg(self.rating_col).alias("avg_user_rating")
        ).join(
            df.groupBy(self.business_col).agg(
                countDistinct(self.user_col).alias("unique_customers"),
                avg("total_reviews").alias("avg_customer_engagement")
            ), on=self.business_col
        )
    
    def _add_competitive_insights(self, df: DataFrame) -> DataFrame:
        """Generate competitive benchmarking metrics"""
        window = Window.orderBy(col("yearly_avg_rating").desc())
        return df.withColumn("competitor_rank", rank().over(window))