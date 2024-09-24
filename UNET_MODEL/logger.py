import logging
import os

def setup_logger(log_filename):
    """
    Setup logger for logging messages to both file and console.
    
    Args:
        log_filename (str): Name of the log file.
    
    Returns:
        logger (logging.Logger): Configured logger.
    """
    # Create logger
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.DEBUG)  # Log level can be adjusted

    # Create a file handler to log to a file
    log_path = os.path.join('logs', f"{log_filename}.log")
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler to log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter for logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add the formatter to handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
