# What is artifact for ??

# ✅ It stores the output of the ingestion step
# ✅ It lets the next stage use ingestion results easily
# ✅ It helps maintain clean architecture
# ✅ It follows industry-standard pipeline design

from dataclasses import dataclass
@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path : str
    transformed_test_file_path : str
    transformed_object_file_path:str 

@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact