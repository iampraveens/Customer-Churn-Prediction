import pandas as pd 
import os
import joblib
from xgboost import XGBClassifier
from CustomerChurn import logger
from CustomerChurn.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer class with a ModelTrainerConfig object.
        Args:
            config (ModelTrainerConfig): The configuration object for model training.
        Returns:
            None
        """
        self.config = config
        
    def train(self):
        """
        Trains a model based on the configuration provided.
        
        Retrieves the training and testing data from the specified paths, preprocesses the data, and trains an XGBClassifier
        model based on the configuration provided. The trained model is then saved to the specified root directory with the
        specified model name.
        """
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        X_train = train_data.drop([self.config.target_column], axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        y_test = train_data[[self.config.target_column]]
        
        model = XGBClassifier(colsample_bytree=self.config.colsample_bytree, learning_rate=self.config.learning_rate,
                              max_depth=self.config.max_depth, n_estimators=self.config.n_estimators,
                              subsample=self.config.subsample)
        model.fit(X_train, y_train)
        
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
        logger.info("Model got built successfully")
        