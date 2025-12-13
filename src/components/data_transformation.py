import sys
from pandas import DataFrame
import pandas as pd
import numpy as np

from src.constants import TARGET_COLUMN , SCHEMA_FILE_PATH
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.utils.main_utils import read_yaml_file , save_numpy_array_data, save_object

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
        
    @staticmethod
    def read_csv_data(file_path) -> DataFrame:
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise MyException(e, sys)
        
    # Apply the standardScaling on Numerical features
    def normalize_data(self) -> Pipeline:
        try:
            logging.info("Entering the Fuction to normalize the data")
            
            std = StandardScaler()

            # Extract the numerical columns
            num_columns = self._schema_config['numerical_columns']

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", std, num_columns),
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            
            logging.info("Exited the normailze function")
            return final_pipeline
        
        except Exception as e:
            raise MyException(e, sys) from e


    # Remove the Outliers
    def remove_outliers(self,df):
        df = df[df['person_age'] < 100]
        df = df[df['person_emp_exp'] < 90]
            
        df.reset_index(drop=True, inplace=True)

        return df
    
    # Convert the categrical data into numerical form
    def encode_categorical_data(self,df):
        df = pd.get_dummies(df, drop_first=True)
        return df
    
    # Drop the id column
    def drop_id_column(self, df):
        df.drop(columns = 'id', inplace = True)
        return df

    # Balance the Target column
    def balancing_target_column(self,X,y):
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        
        return X_res,y_res


    # Final pipeline
    def initiate_data_transformation(self) -> DataTransformationArtifact:

        try:
            logging.info("Load the Raw test and train data")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = DataTransformation.read_csv_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataTransformation.read_csv_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Drop the id column
            train_df = self.drop_id_column(train_df)
            test_df = self.drop_id_column(test_df)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")



            logging.info("Removing the outliers")
            logging.info("removing outliers from the train data inputs")
            input_feature_train_df = self.remove_outliers(input_feature_train_df)
            logging.info("removing outliers from the test data inputs")
            input_feature_test_df = self.remove_outliers(input_feature_test_df)


            logging.info("Encode the Categorical data")
            logging.info("Encode Training data")
            input_feature_train_df = self.encode_categorical_data(input_feature_train_df)
            logging.info("Encode the Test data")
            input_feature_test_df = self.encode_categorical_data(input_feature_test_df)


            logging.info("Start Normalizing the Numerical data in train-test df")
            preprocessor = self.normalize_data()
            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")


            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            logging.info("Bbalncing classes if Train data")
            input_feature_train_final, target_feature_train_final = self.balancing_target_column(
                input_feature_train_arr, target_feature_train_df
            )
            logging.info("Balancing classes of test data")
            input_feature_test_final, target_feature_test_final = self.balancing_target_column(
                input_feature_test_arr, target_feature_test_df
            )

            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys)

