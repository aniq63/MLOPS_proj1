# from src.logger import logging

# logging.debug("This is a debug message.")
# logging.info("This is an info message.")
# logging.warning("This is a warning message.")
# logging.error("This is an error message.")
# logging.critical("This is a critical message.")

# from src.exception import MyException
# from src.logger import logging
# import sys

# try:
#     num = 9 + 'N'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e
from src.pipline.training_pipeline import TrainPipeline
pipeline = TrainPipeline()
pipeline.run_pipeline()
