import logging
import os

def setup_logger(file_path):
    base_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    
    logger = logging.getLogger(file_name_without_extension)
    logger.setLevel(logging.INFO) 
    
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    
    return logger
