import logging
import os
from datetime import datetime


class Logger:
    """
    Practical logger for tracking smart home model operations.
    Logs to both console and a file with timestamps and log levels.
    """
    
    def __init__(self, name="SmartHomeModel", log_file=None, level=logging.INFO):
        """
        Initialize the logger with basic settings.
        
        Args:
            name (str): Logger name
            log_file (str): Path to log file (if None, creates a timestamped file)
            level (int): Logging level (INFO, DEBUG, etc.)
        """
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Format with timestamp, level, and message
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (with timestamp if no filename provided)
        if log_file is None:
            # Create logs directory if it doesn't exist
            if not os.path.exists('logs'):
                os.makedirs('logs')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.info(f"Logger initialized. Logging to console and {log_file}")
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    def exception(self, message):
        """Log exception with traceback"""
        self.logger.exception(message)