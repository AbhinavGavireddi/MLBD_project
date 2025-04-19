# End-to-End NLP Pipeline for Yelp Reviews

---

## Overview
This project provides a modular, reproducible, and extensible NLP pipeline for extracting structured insights from the Yelp Open Dataset. It covers every stage from raw data ingestion to advanced analytics, including:

- Aspect-based sentiment analysis (ABSA)
- Named Entity Recognition (NER)
- Topic modeling
- Sentiment classification
- Automated evaluation and visualizations

The pipeline is built entirely with open-source tools and is suitable for both research and production use. Every component is documented in detail for users of all backgrounds, including novices.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Testing](#testing)
- [Key Files and Their Roles](#key-files-and-their-roles)
- [Evaluation Metrics: Definitions & Interpretations](#evaluation-metrics-definitions--interpretations)
- [Pipeline Flow](#pipeline-flow)
- [Detailed Module & Function Reference](#detailed-module--function-reference)
- [Models & Methods Used](#models--methods-used)
- [Troubleshooting & Best Practices](#troubleshooting--best-practices)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
---

## Project Structure

```
project/
├── README.md
├── LICENSE
├── requirements.txt
├── config/
│   └── config.yaml                 # All pipeline config (paths, hyperparams)
├── src/
│   ├── data_ingest.py              # Data loading and saving
│   ├── preprocess.py               # Text cleaning and tokenization
│   ├── ner.py                      # Named Entity Recognition
│   ├── absa.py                     # Aspect-Based Sentiment Analysis
│   ├── topic_model.py              # Topic Modeling (LDA/NMF)
│   ├── sentiment.py                # Sentiment Classification (DistilBERT)
│   └── evaluate.py                 # Evaluation metrics for all tasks
├── main.py                         # Orchestrates the full pipeline
├── download_data.py                # Script to download sample/raw data
├── tests/                          # Unit and integration tests
│   ├── test_preprocess.py
│   ├── test_sentiment.py
│   ├── test_integration.py
├── notebooks/                      # Jupyter notebooks for EDA
│   └── exploratory_analysis.ipynb
├── data/
│   ├── raw/                        # Place raw Yelp data here
│   └── processed/                  # Pipeline outputs
├── models/                         # Model checkpoints (optional)
├── logs/                           # Log files
├── reports/                        # Evaluation and topic keywords
└── img/                            # Visualizations
```

---

---

## Setup & Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/extract-yelp-info.git
    cd extract-yelp-info
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    pip install pyabsa transformers torch gensim spacy matplotlib seaborn scikit-learn pandas
    ```
3. **Download Data**
    ```bash
    python download_data.py
    # Or manually download the full Yelp Open Dataset and place in data/raw/
    ```

---

## Usage

### Run the Entire Pipeline

```bash
python main.py
```
- All processed data, reports, and images will be saved to the appropriate folders.

### Run Individual Steps

Each script in `src/` can be run with `--help` for CLI options. For example:

```bash
python src/absa.py --input data/processed/reviews_cleaned.csv --output data/processed/reviews_absa.csv --batch_size 32
```

### Evaluate Topic Coherence

```bash
python src/evaluate.py --task topic \
  --cleaned_reviews data/processed/reviews_cleaned.csv \
  --topic_keywords reports/topics_keywords.csv \
  --report reports/topic_coherence.json
```

---

## Testing

1. **Run all tests**
    ```bash
    pip install pytest
    pytest tests/
    ```
2. **Test Coverage**
    - Unit tests for preprocessing, sentiment, and integration.
    - Add your own for ABSA, NER, topic modeling as needed.

---


## Key Files and Their Roles

### main.py
- **Purpose:** Orchestrates the entire NLP pipeline in the correct order and generates all visualizations.
- **How it works:** Calls each module’s main function, checks for errors, and saves plots to `img/`.
- **Role:** The single entrypoint for running the whole workflow.

### download_data.py
- **Purpose:** Downloads a sample Yelp review dataset for demo/testing.
- **How it works:** Fetches a small review JSON from a public mirror. For full data, instructs manual download.
- **Role:** Eases onboarding and testing without the full Yelp dataset.

### src/data_ingest.py
- **Purpose:** Loads raw Yelp JSON/CSV files and saves them as DataFrames for downstream processing.
- **Key Functions:**
  - `load_reviews(json_path)`: Loads reviews JSON → DataFrame.
  - `load_business(json_path)`: Loads business JSON → DataFrame.
- **Role:** Ensures data is in a tabular format for the pipeline.

### src/preprocess.py
- **Purpose:** Cleans and tokenizes review text.
- **Key Functions:**
  - `clean_text(text)`: Lowercases, removes special chars, normalizes whitespace.
  - `preprocess_text(text)`: Tokenizes and lemmatizes the text.
  - `process_reviews(input_csv, output_csv, ...)`: Applies cleaning/tokenization to all reviews.
- **Role:** Standardizes text for all downstream NLP tasks.

### src/ner.py
- **Purpose:** Extracts named entities (e.g., locations, orgs) using spaCy.
- **Key Functions:**
  - `extract_entities(text)`: Returns a list of entities for a review.
  - `process_reviews(input_csv, output_csv, ...)`: Applies NER to all reviews and saves results.
- **Role:** Identifies key entities for further analysis or filtering.

### src/absa.py
- **Purpose:** Performs Aspect-Based Sentiment Analysis (ABSA) using PyABSA.
- **Key Functions:**
  - `analyze_aspects(texts, aspects, ...)`: Extracts aspect terms and their sentiment from reviews.
  - `process_reviews(input_csv, output_csv, aspects, ...)`: Applies ABSA to all reviews for predefined aspects (e.g., food, service).
- **Role:** Reveals fine-grained sentiment for specific business aspects.
- **Model:** PyABSA ATEPC (Aspect Term Extraction & Polarity Classification)

### src/topic_model.py
- **Purpose:** Discovers topics in reviews using LDA or NMF.
- **Key Functions:**
  - `run_topic_model(...)`: Trains topic model, outputs topic distributions per review and top keywords per topic.
- **Role:** Reveals main themes in the dataset for exploration or clustering.
- **Models:**
  - LDA (Latent Dirichlet Allocation)
  - NMF (Non-negative Matrix Factorization)

### src/sentiment.py
- **Purpose:** Classifies overall sentiment of reviews using DistilBERT.
- **Key Functions:**
  - `classify_sentiment(texts, model_path, batch_size)`: Predicts sentiment for a batch of reviews.
  - `process_reviews(input_csv, output_csv, ...)`: Classifies sentiment for all reviews and saves results.
- **Role:** Provides a high-level sentiment label for each review.
- **Model:** Hugging Face Transformers DistilBERT (fine-tuned for sentiment)

### src/evaluate.py
- **Purpose:** Evaluates classification, ABSA, and topic modeling outputs.
- **Key Functions:**
  - `classification_metrics(y_true, y_pred)`: Computes precision, recall, F1-score.
  - `evaluate_classification(pred_path, gt_path, ...)`: Evaluates predictions vs. ground truth.
  - `evaluate_topic_coherence(cleaned_reviews_path, topic_keywords_path, ...)`: Computes topic coherence (c_v, u_mass) using gensim.
- **Role:** Quantifies model performance and topic interpretability.
- **How to use topic coherence:**
    ```bash
    python src/evaluate.py --task topic \
      --cleaned_reviews data/processed/reviews_cleaned.csv \
      --topic_keywords reports/topics_keywords.csv \
      --report reports/topic_coherence.json
    ```
---

## Pipeline Flow

1. **Download Data:** Run `download_data.py` to get a sample dataset (or place full Yelp data in `data/raw/`).
2. **Run Pipeline:** Execute `main.py` to run all steps in order: ingestion → preprocessing → NER → ABSA → topic modeling → sentiment → visualizations.
3. **Outputs:** Processed CSVs in `data/processed/`, reports in `reports/`, and visualizations in `img/`.
4. **Evaluation:** Use `src/evaluate.py` for metrics on classification or topic coherence.

---

## Detailed Module & Function Reference

### data_ingest.py
- `load_reviews(json_path)`: Loads Yelp reviews JSON into a DataFrame.
- `load_business(json_path)`: Loads Yelp business JSON into a DataFrame.

### preprocess.py
- `clean_text(text)`: Cleans a string (lowercase, remove special chars).
- `preprocess_text(text)`: Tokenizes and lemmatizes a string.
- `process_reviews(input_csv, output_csv, ...)`: Processes and saves cleaned/tokenized reviews.

### ner.py
- `extract_entities(text)`: Extracts named entities using spaCy.
- `process_reviews(input_csv, output_csv, ...)`: Applies NER to all reviews and saves to CSV.

### absa.py
- `analyze_aspects(texts, aspects, model_path, batch_size)`: Extracts aspect terms and their sentiment using PyABSA.
- `process_reviews(input_csv, output_csv, aspects, ...)`: Runs ABSA for each review and saves aspect sentiment per aspect.

### topic_model.py
- `run_topic_model(input_path, output_dist, output_keywords, ...)`: Fits topic model (LDA/NMF), saves topic distributions and keywords.

### sentiment.py
- `classify_sentiment(texts, model_path, batch_size)`: Predicts sentiment for a batch of reviews using DistilBERT.
- `process_reviews(input_csv, output_csv, ...)`: Classifies sentiment for all reviews and saves results.

### evaluate.py
- `classification_metrics(y_true, y_pred)`: Computes precision, recall, F1-score.
- `evaluate_classification(pred_path, gt_path, ...)`: Evaluates predictions vs. ground truth.
- `evaluate_topic_coherence(cleaned_reviews_path, topic_keywords_path, ...)`: Computes topic coherence (c_v, u_mass) using gensim.

### main.py
- `run_pipeline()`: Runs the full pipeline in order, checks for errors.
- `make_visualizations()`: Generates and saves plots (review length, sentiment, aspect sentiment, topic keywords).

---

## Models & Methods Used

- **spaCy:** Named Entity Recognition (pretrained English model)
- **PyABSA:** Aspect-based sentiment analysis (ATEPC model)
- **scikit-learn:** LDA, NMF for topic modeling; metrics for evaluation
- **Hugging Face Transformers:** DistilBERT for sentiment classification
- **gensim:** Topic coherence metrics (c_v, u_mass)
- **matplotlib, seaborn:** Visualizations
- **pandas:** Data manipulation throughout

---

## Evaluation Metrics: Definitions & Interpretations

### Classification Metrics

- **Precision**: The proportion of predicted positive labels that are actually correct. High precision means that when the model predicts a certain class (e.g., "positive" sentiment), it is usually right.
    - Formula: `Precision = TP / (TP + FP)`
    - TP = True Positives, FP = False Positives
- **Recall**: The proportion of actual positive labels that were correctly identified by the model. High recall means the model finds most of the relevant items.
    - Formula: `Recall = TP / (TP + FN)`
    - FN = False Negatives
- **F1-Score**: The harmonic mean of precision and recall. It balances both metrics and is especially useful when classes are imbalanced.
    - Formula: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
- **Weighted Average**: All metrics are reported as weighted averages, meaning each class's metric is weighted by its support (number of true instances for each class).

### Topic Modeling Metrics

- **Topic Coherence (c_v)**: Measures the semantic similarity between high-scoring words in a topic. Higher values (closer to 1) indicate more interpretable and meaningful topics. Computed using word co-occurrence in the original texts.
    - Range: Typically 0 (worst) to 1 (best)
    - Used for: Comparing different topic models or tuning the number of topics.
- **Topic Coherence (u_mass)**: Based on document co-occurrence counts, this metric can be negative. Higher (closer to 0) is better, but it is less interpretable than c_v and more sensitive to preprocessing and data sparsity.
    - Range: Negative values, higher is better (closer to 0)

### When to Use Each Metric

- Use **precision** when false positives are costly (e.g., marking a negative review as positive).
- Use **recall** when missing positives is costly (e.g., missing a negative review).
- Use **F1-score** for balanced performance, especially with class imbalance.
- Use **topic coherence** to select the best topic model or number of topics for interpretability.

---

## Troubleshooting & Best Practices

- **Missing Data/Files:** Ensure you have run `download_data.py` or placed the full dataset in `data/raw/`.
- **Out of Memory:** For large datasets, reduce batch size in ABSA or sentiment scripts.
- **Reproducibility:** Set seeds in `config/config.yaml` for deterministic runs.
- **Logging:** All scripts log progress and errors; check the `logs/` folder for details.
- **Error Handling:** The pipeline skips steps with missing data and logs errors without crashing.
- **Visualization Failures:** All plots are defensively coded to skip if required data is missing.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Yelp Open Dataset](https://www.yelp.com/dataset)
- [PyABSA](https://github.com/yangheng95/PyABSA)
- [spaCy](https://spacy.io/)
- [scikit-learn](https://scikit-learn.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [gensim](https://radimrehurek.com/gensim/)
- Inspired by [SF Brigade README template](https://github.com/sfbrigade/standard-readme-template)


