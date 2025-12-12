import os
import sys
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from sklearn.model_selection import train_test_split

class DataIngestion:
      def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
            
            # Load the dataingestion configs
            # Means their paths and parameters e.t.c
            try:
                  self.data_ingestion_config = data_ingestion_config
            except Exception as e:
                  raise MyException(e,sys)
      
      ## Export the Raw data from the Database
      def export_data_into_feature_store(self) -> DataFrame:
            try:
                # make a connection with db using dataacces file that use configuration folder
                logging.info("Make a connection with MongoDB")
                my_data = Proj1Data()
                logging.info("Extract the data from MongoDB")
                
                # Extract the data using the Proj1Data class that is also converted the in to dataframe
                dataframe = my_data.export_collection_as_dataframe(collection_name = self.data_ingestion_config.collection_name)
                logging.info(f"Shape of Dataset is {dataframe.shape}")

                # now make a directory where we want to store our raw data in form of csv
                feature_store_file_path = self.data_ingestion_config.feature_store_file_path
                dir_path = os.path.dirname(feature_store_file_path)
                os.makedirs(dir_path , exist_ok=True)
                logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")

                # Convert our dataframe into the csv and store in feature store file path as csv
                dataframe.to_csv(feature_store_file_path, index= False, header= True)

                return dataframe

            except Exception as e:
                  raise MyException(e,sys) from e 
            
      def split_data_as_train_test(self,dataframe: DataFrame) ->None:
            logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

            try:
                  # Make a directory for to store the test and train data
                  training_file_path = os.path.dirname(self.data_ingestion_config.training_file_path)
                  os.makedirs(training_file_path, exist_ok=True)

                  train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
                  logging.info("Performed train test split on the dataframe")
                  logging.info(
                  "Exited split_data_as_train_test method of Data_Ingestion class"
                  )

                  logging.info(f"Exporting train and test file path.")
                  train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
                  test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
  
                  logging.info(f"Exported train and test file path.")

            except Exception as e:
                  raise MyException(e, sys) from e
       
      def initiate_data_ingestion(self) ->DataIngestionArtifact:
        try:
            df = self.export_data_into_feature_store()
            self.split_data_as_train_test(df)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
             raise MyException(e, sys) from e
                  
   

                  
            