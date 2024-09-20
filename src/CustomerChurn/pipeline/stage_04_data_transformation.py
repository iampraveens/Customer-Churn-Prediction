from CustomerChurn.config.configuration import ConfigurationManager
from CustomerChurn.components.data_transformation import DataTransformation
from CustomerChurn import logger
from pathlib import Path



STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        """
        Initializes an instance of the DataTransformationTrainingPipeline class.
        
        This is a special method that Python calls when an object is instantiated from the class.
        """
        pass

    def main(self):
        
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data = data_transformation.load_data()
            data = data_transformation.encode_data(data=data)
            data = data_transformation.feature_engineering(data=data)
            balanced_data = data_transformation.data_balancing(data=data, method='SMOTE')
            data_transformation.train_test_splitting(data=balanced_data)

        except Exception as e:
            print(e)