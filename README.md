# End-to-End NLP Pipeline for Yelp Reviews

---

## Overview
This project provides a scalable, modular NLP pipeline for extracting structured insights from the Yelp Reviews Open Dataset. It covers every stage from raw data ingestion to business intelligence reporting, including:

- Data ingestion and storage (PySpark, Parquet)
- Text preprocessing (cleaning, tokenization, stopword removal, n-grams)
- Feature extraction (TF-IDF, sentiment, readability, emotion, keyphrase extraction)
- Business analytics (temporal trends, aspect analysis, customer behavior, benchmarking)
- Reporting (automated report generation)

---

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml                 # All pipeline config (paths, hyperparams)
├── src/
│   ├── __init__.py                 # Logging, utility setup
│   ├── data_extractor.py           # Data ingestion and saving
│   ├── preprocessor.py             # Text cleaning and tokenization
│   ├── feature_extraction.py       # Feature engineering (TF-IDF, sentiment, readability, emotion, keyphrases)
│   ├── business_metrics_calculator.py # Business analytics (metrics, trends, benchmarking)
│   ├── report_generator.py         # Automated report generation
│   ├── utils.py                    # Config loading, Spark session helpers
├── main.py                         # Pipeline orchestrator (run end-to-end)
├── notebooks/                      # Jupyter notebooks for EDA, prototyping
│   ├── exploratory_analysis.ipynb
│   └── research_pipeline.py
├── data/
│   ├── raw/                        # Place raw Yelp data here
│   └── processed/                  # Pipeline outputs (Parquet)
├── models/                         # Model checkpoints (optional)
├── logs/                           # Log files
├── reports/                        # Evaluation and topic keywords
└── img/                            # Visualizations
```

---

## Setup & Installation

1. **Clone the repository**
    ```bash
    git clone <https://github.com/AbhinavGavireddi/MLBD_project.git>
    cd <MLBD_project>
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    # Additional dependencies (if needed):
    # pip install pyspark rake-nltk pyyaml loguru
    ```
3. **Download Data**
    - Place the Yelp review dataset in `data/raw/` as specified in `config/config.yaml`.

---

## Usage

### Run the Entire Pipeline

```bash
python main.py
```
- All processed data, reports, and images will be saved to the appropriate folders.

### Run Individual Modules (for development/testing)

Each script in `src/` can be imported and run interactively, or you can test individual modules via the `tests/` directory.

---

## Key Files and Their Roles

- **main.py**: Orchestrates the pipeline (data ingestion → preprocessing → feature extraction → analytics → reporting).
- **src/data_extractor.py**: Loads raw Yelp data, saves processed DataFrames as Parquet.
- **src/preprocessor.py**: Cleans and tokenizes text, removes stopwords, generates n-grams.
- **src/feature_extraction.py**: Extracts TF-IDF, sentiment, readability, emotion, and keyphrases.
- **src/business_metrics_calculator.py**: Computes business metrics (trends, aspects, customer behavior, benchmarking).
- **src/report_generator.py**: Generates and saves business intelligence reports.
- **src/utils.py**: Loads config (YAML), manages Spark session.
- **src/__init__.py**: Sets up logging and utility functions.
- **tests/**: Unit and integration tests for core modules.

---

## Models & Methods Used

- **PySpark**: Distributed data processing, feature engineering
- **TF-IDF**: Text vectorization (CountVectorizer, IDF)
- **RAKE (rake_nltk)**: Keyphrase extraction
- **Flesch-Kincaid**: Readability scoring
- **Keyword-based Sentiment & Emotion**: Simple heuristics for demonstration
- **Window Functions**: Temporal analysis
- **Parquet**: Efficient columnar storage
- **Loguru**: Logging
- **YAML**: Config management

---

## Evaluation & Testing

- **Metrics**: Precision, recall, F1-score, topic coherence, readability
- **Reproducibility**: All configs in YAML, random seeds set in `config/config.yaml`
