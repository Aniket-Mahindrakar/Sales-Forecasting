import yaml

from utils.mlflow_utils import MLflowManager
from feature_engineering.feature_pipeline import FeatureEngineer


class ModelTrainer:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config["models"]
        self.training_config = self.config["training"]