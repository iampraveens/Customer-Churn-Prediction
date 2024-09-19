from CustomerChurn.config.configuration import ConfigurationManager
from CustomerChurn.components.data_preprocessing import DataPreprocessing
from CustomerChurn import logger

STAGE_NAME = "Data Preprocessing stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        """
        Initializes the DataPreprocessingTrainingPipeline class.
        
        This is a special method that Python calls when an object is instantiated from the class.
        """
        pass

    def main(self):
        """
        This is the main method of the DataPreprocessingTrainingPipeline class.
        
        It is responsible for orchestrating the data preprocessing process by creating 
        a ConfigurationManager object, retrieving the data preprocessing configuration, 
        creating a DataPreprocessing object, and calling its data_cleaning method.
        """
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.data_cleaning()
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        
        