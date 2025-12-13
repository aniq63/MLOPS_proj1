import sys
import os

from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.constants import TARGET_COLUMN

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_training_config: ModelTrainingConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            # keep internal name used across this class
            self.model_trainer_config = model_training_config
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_training(self):
        logging.info("Enter Model training Class")
        try:
            logging.info("Load the transformed training data")
            df = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)

            logging.info("Split the training data into train and evaluation sets")
            X = df[:,:-1]
            y = df[:,-1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            logging.info("Initialize the Random Forest Model")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            logging.info("Model training started")
            model.fit(X_train, y_train)
            logging.info("Model training completed")

            logging.info("Start prediction on evaluation data")
            y_pred = model.predict(X_test)
            logging.info("Prediction completed")

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            logging.info(f"Model Performance - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

            # Check if accuracy meets the expected threshold
            base_accuracy = self.model_trainer_config.trained_model_expected_score
            
            if accuracy < base_accuracy:
                logging.warning(f"Accuracy {accuracy:.4f} is less than base accuracy {base_accuracy:.4f}. Trying Gradient Boosting model...")
                
                # Try Gradient Boosting as an alternative model
                logging.info("Initialize Gradient Boosting Model")
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                logging.info("Gradient Boosting model training started")
                model.fit(X_train, y_train)
                logging.info("Gradient Boosting model training completed")
                
                logging.info("Prediction with Gradient Boosting model")
                y_pred = model.predict(X_test)
                
                # Recalculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                
                logging.info(f"Gradient Boosting Performance - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                
                if accuracy < base_accuracy:
                    raise MyException(
                        f"No model achieved the expected accuracy {base_accuracy:.4f}. "
                        f"Best achieved: {accuracy:.4f}",
                        sys
                    )

            # Create artifact directory
            logging.info("Creating model trainer artifact directory")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            
            # Save the trained model
            logging.info(f"Saving trained model to {self.model_trainer_config.trained_model_file_path}")
            save_object(self.model_trainer_config.trained_model_file_path, model)
            
            # Create metric artifact
            metric_artifact = ClassificationMetricArtifact(
                f1_score=accuracy,  # Using accuracy as f1_score placeholder
                precision_score=precision,
                recall_score=recall
            )
            
            # Create and return ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )
            
            logging.info("Model training completed successfully")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
