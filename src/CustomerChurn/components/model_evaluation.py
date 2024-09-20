from pathlib import Path    
import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score, recall_score
import joblib
from CustomerChurn.entity.config_entity import ModelEvaluationConfig
from CustomerChurn.utils.common import save_json
from CustomerChurn import logger

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        """
        Evaluate the model on the given actual and predicted values.

        Parameters
        ----------
        actual : array-like
            Actual values.
        pred : array-like
            Predicted values.

        Returns
        -------
        tuple
            A tuple containing accuracy, f1 score and recall score.
        """
        accuracy = accuracy_score(actual, pred)
        f1_Score = f1_score(actual, pred)
        recall_Score = recall_score(actual, pred)
        return accuracy, f1_Score, recall_Score
    
    def save_results(self):

        """
        Evaluate the model and save the evaluation metrics to a JSON file.

        Saves the accuracy, f1 score and recall score to a JSON file at the path specified in the configuration.
        """
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[[self.config.target_column]]
        
        predicted = model.predict(X_test)

        (accuracy, f1_Score, recall_Score) = self.eval_metrics(y_test, predicted)
        
        # Saving metrics as local
        scores = {"accuracy": accuracy, "f1_Score": f1_Score, "recall_Score": recall_Score}
        save_json(path=Path(self.config.metric_file_name), data=scores)
        logger.info("Metric saved successfully")