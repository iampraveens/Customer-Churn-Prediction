import os
from pathlib import Path
import urllib.request as request
import zipfile
from CustomerChurn import logger
from CustomerChurn.utils.common import get_size
from CustomerChurn.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
        Initializes the DataIngestion class with a DataIngestionConfig object.
        Args:
            config (DataIngestionConfig): The configuration object for data ingestion.
        Returns:
            None
        """
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with a DataIngestionConfig object.
        Args:
            config (DataIngestionConfig): The configuration object for data ingestion.
        Returns:
            None
        """
        self.config = config
        
        
    def download_file(self):
        """
        Downloads a file from a specified URL and saves it to a local directory.
        """
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url= self.config.source_URL,
                filename= self.config.local_data_file,
            )
            logger.info(f"{filename} downloaded with the following info: \n{headers}")
        else:
            logger.info(f"file already exists of size: {get_size(Path(self.config.local_data_file))}")
            
    
    def extract_zip_file(self):
        """
        Extracts a zip file to a specified directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)