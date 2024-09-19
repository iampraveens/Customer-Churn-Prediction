import os
import pandas as pd
from CustomerChurn import logger
from CustomerChurn.entity.config_entity import DataValidationConfig

class DataValidation:
    """
        Initializes the DataValidation class with a DataValidationConfig object.
        
        Args:
            config (DataValidationConfig): The configuration object for data validation.
        """
    def __init__(self, config: DataValidationConfig):
        """
        Initializes the DataValidation class with a DataValidationConfig object.
        
        Args:
            config (DataValidationConfig): The configuration object for data validation.
        """
        self.config = config
        
    def validate_all_columns(self)-> bool:
        """
        Validates all columns in the dataset against the schema.
        This function checks if all columns in the dataset are present in the schema.
        If any column is missing, it sets the validation status to False and writes it to the status file.
        If all columns are present, it sets the validation status to True and writes it to the status file.
        Args:
            None
        Returns:
            bool: The validation status of the dataset.
        """
        try:
            validation_status = None
            
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            
            all_schemas = self.config.all_schema.keys()
            
            for col in all_cols:
                if col not in all_schemas:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"validation status: {validation_status}")
                        
            return validation_status
        
        except Exception as e:
            raise e