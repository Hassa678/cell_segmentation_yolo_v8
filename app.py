import sys
from cellSegmentation_v8.logger import logging
from cellSegmentation_v8.exception import AppException
logging.info("app")

try:
    pass
    #a = 4/'6'
except Exception as e:
    
    raise AppException(e,sys)