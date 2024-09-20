import os
import pandas as pd
from typing import Union
from CustomerChurn import logger
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from CustomerChurn.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation class with a DataTransformationConfig object.
        Args:
            config (DataTransformationConfig): The configuration object for data transformation.
        """
        self.config = config
    
    def load_data(self) -> pd.DataFrame:
        """
        Loads the data from the specified path and returns a pandas DataFrame.

        Args:
            None

        Returns:
            pd.DataFrame: The loaded data.
        """
        try:
            data = pd.read_csv(self.config.data_path)
            logger.info("Data loaded successfully.")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e
        
    def encode_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes the categorical columns in the given DataFrame as integers.
        Binary columns are replaced with 0 or 1, and categorical columns are one-hot encoded.
        
        Args:
            data (pd.DataFrame): The DataFrame to encode.
            
        Returns:
            pd.DataFrame: The encoded DataFrame.
        """
        try:
            binary_columns = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn', 'PhoneService']
            data[binary_columns] = data[binary_columns].applymap(lambda x: 1 if x == 'Yes' else 0)
            
            data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Female' else 0)

            data['MultipleLines'] = data['MultipleLines'].map({'No phone service': 0, 'No': 0, 'Yes': 1})

            internet_service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            data[internet_service_columns] = data[internet_service_columns].replace({'No internet service': 0, 'No': 0, 'Yes': 1})

            categorical_columns = ['InternetService', 'Contract', 'PaymentMethod']
            data = pd.get_dummies(data, columns=categorical_columns, drop_first=True, dtype='int')

            return data

        except Exception as e:
            logger.error(f"Error during data encoding: {str(e)}")
            raise e

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a new column 'TotalCharges_per_month' by dividing TotalCharges by (tenure + 1),
        effectively calculating the average monthly charges for each customer.
        
        Args:
            data (pd.DataFrame): The DataFrame to feature engineer.
            
        Returns:
            pd.DataFrame: The feature engineered DataFrame.
        """
        try:
            data['TotalCharges_per_month'] = data['TotalCharges'] / (data['tenure'] + 1)  # Avoiding division by zero
            logger.info("Feature engineering complete.")
            return data
        
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise e

    def data_balancing(self, data: pd.DataFrame, method: str) -> Union[pd.DataFrame, pd.Series]:
        """
        Balances the given DataFrame using the specified method.

        Args:
            data (pd.DataFrame): The DataFrame to balance.
            method (str): The balancing method to use. Possible values are 'SMOTE' and 'SMOTEENN'.
            
        Returns:
            pd.DataFrame or pd.Series: The balanced DataFrame or Series.
        """
        try:
            X = data.drop('Churn', axis=1)
            y = data['Churn']
            
            if method == 'SMOTE':
                smote = SMOTE()
                X_res, y_res = smote.fit_resample(X, y)
                logger.info("Data balanced using SMOTE.")
            elif method == 'SMOTEENN':
                smoteenn = SMOTEENN()
                X_res, y_res = smoteenn.fit_resample(X, y)
                logger.info("Data balanced using SMOTEENN.")
            else:
                raise ValueError(f"Invalid method: {method}. Choose either 'SMOTE' or 'SMOTEENN'.")
            
            logger.info(f"Before balancing: {y.value_counts()}")
            logger.info(f"After balancing: {y_res.value_counts()}")
            
            balanced_data = pd.DataFrame(X_res)
            balanced_data['Churn'] = y_res.values
            return balanced_data

        except Exception as e:
            logger.error(f"Error during data balancing: {str(e)}")
            raise e

    def train_test_splitting(self, data: pd.DataFrame):
        
        """
        Splits the given DataFrame into training and test sets using train_test_split from scikit-learn.

        Args:
            data (pd.DataFrame): The DataFrame to split.

        Returns:
            None
        """
        try:
            train, test = train_test_split(data)

            if isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame):
                train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
                test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

                logger.info("Data split into training and test sets.")
                logger.info(f"Train set shape: {train.shape}")
                logger.info(f"Test set shape: {test.shape}")
            else:
                raise ValueError("Train-test split did not return DataFrames.")

        except Exception as e:
            logger.error(f"Error during train-test splitting: {str(e)}")
            raise e
        