from CustomerChurn.constants import *
from CustomerChurn.utils.common import read_yaml, create_directories
from CustomerChurn.entity.config_entity import (DataIngestionConfig, DataValidationConfig,
                                                DataPreprocessingConfig, DataTransformationConfig)


class ConfigurationManager:
    """
        This is the constructor method for the ConfigurationManager class.
        
        It initializes the ConfigurationManager object with the provided configuration, 
        parameters, and schema file paths. It reads the YAML files and stores their 
        contents in the config, params, and schema attributes respectively. It also 
        creates the directories specified in the artifacts_root attribute of the config.
        
        Parameters:
            config_filepath (str): The path to the configuration YAML file.
            params_filepath (str): The path to the parameters YAML file.
            schema_filepath (str): The path to the schema YAML file.
        
        Returns:
            None
        """
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config.artifacts_root])
        
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves the data ingestion configuration from the ConfigurationManager object.
        
        This method reads the data ingestion configuration from the config attribute, 
        creates the root directory specified in the configuration if it does not exist, 
        and returns a DataIngestionConfig object containing the configuration details.
        
        Returns:
            DataIngestionConfig: The data ingestion configuration object.
        """
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Retrieves the data validation configuration from the ConfigurationManager object.
        
        This method reads the data validation configuration from the config attribute, 
        creates the root directory specified in the configuration if it does not exist, 
        and returns a DataValidationConfig object containing the configuration details.
        
        Returns:
            DataValidationConfig: The data validation configuration object.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir = config.root_dir,
            unzip_data_dir = config.unzip_data_dir,
            STATUS_FILE = config.STATUS_FILE,
            all_schema=schema
        )
        
        return data_validation_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        Retrieves the data preprocessing configuration from the ConfigurationManager object.
        
        This method reads the data preprocessing configuration from the config attribute, 
        creates the root directory specified in the configuration if it does not exist, 
        and returns a DataPreprocessingConfig object containing the configuration details.
        
        Returns:
            DataPreprocessingConfig: The data preprocessing configuration object.
        """
        config = self.config.data_preprocessing
        
        create_directories([config.root_dir])
        
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir = config.root_dir,
            data_path = config.data_path
        )
        
        return data_preprocessing_config
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        
        """
        Retrieves the data transformation configuration from the ConfigurationManager object.
        
        This method reads the data transformation configuration from the config attribute, 
        creates the root directory specified in the configuration if it does not exist, 
        and returns a DataTransformationConfig object containing the configuration details.
        
        Returns:
            DataTransformationConfig: The data transformation configuration object.
        """
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config