from loguru import logger as loguru_logger
import os
import yaml
from threading import Lock

class LoggerSingleton:
    _instance = None
    _lock = Lock()

    def __new__(cls, log_dir="logs", log_file="pipeline.log", log_level="INFO"):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_dir="logs", log_file="pipeline.log", log_level="INFO"):
        if self._initialized:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        loguru_logger.remove()
        loguru_logger.add(log_path, level=log_level, format="{time} {level} {message}", rotation="10 MB", retention="10 days")
        loguru_logger.add(lambda msg: print(msg, end=""), level=log_level, format="{time} {level} {message}")
        loguru_logger.info(f"Loguru singleton logger initialized. Logs will be saved to {log_path}")
        self._initialized = True
        self.logger = loguru_logger

    @staticmethod
    def get_logger():
        return LoggerSingleton().logger

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    logger = LoggerSingleton.get_logger()
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        # Return default configuration
        return {
            'random_seed': 42,
            'raw_data_path': 'data/raw/',
            'processed_data_path': 'data/processed/',
            'models_path': 'models/',
            'logs_path': 'logs/',
            'reports_path': 'reports/',
            'absa': {'aspects': ["food", "service", "price", "ambience"]},
            'lda': {'n_topics': 10, 'max_iter': 10},
            'distilbert': {'pretrained_model': 'distilbert-base-uncased', 'epochs': 3, 'batch_size': 16}
        }

def ensure_dirs():
    """Create necessary directories if they don't exist."""
    dirs = ["data/raw", "data/processed", "models", "logs", "reports", "img"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# Convenience function for pipeline entry points
setup_logging = LoggerSingleton
