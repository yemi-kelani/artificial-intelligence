"""
Logging configuration for DQN TicTacToe v2.
Provides centralized logging with different levels and formatters.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "DQN_TicTacToe",
    level: int = logging.INFO,
    log_dir: str = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and/or file handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (created if doesn't exist)
        console_output: Whether to output to console
        file_output: Whether to output to file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        if log_dir is None:
            # Default to logs directory in project root
            from models.common import get_root_directory
            log_dir = os.path.join(get_root_directory(), "logs", "DQN_TicTacToe_v2")
        
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Also create a symlink to latest log
        latest_log = os.path.join(log_dir, f"{name}_latest.log")
        if os.path.exists(latest_log):
            os.remove(latest_log)
        try:
            os.symlink(log_file, latest_log)
        except (OSError, NotImplementedError):
            # Symlinks might not work on all systems
            pass
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name (defaults to module name)
        
    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'DQN_TicTacToe')
        else:
            name = 'DQN_TicTacToe'
    
    # Check if logger already exists
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)
    
    # Create new logger with default settings
    return setup_logger(name)


# Convenience functions for different log levels
def log_debug(message: str, logger_name: str = None):
    """Log a debug message."""
    get_logger(logger_name).debug(message)


def log_info(message: str, logger_name: str = None):
    """Log an info message."""
    get_logger(logger_name).info(message)


def log_warning(message: str, logger_name: str = None):
    """Log a warning message."""
    get_logger(logger_name).warning(message)


def log_error(message: str, logger_name: str = None):
    """Log an error message."""
    get_logger(logger_name).error(message)


def log_critical(message: str, logger_name: str = None):
    """Log a critical message."""
    get_logger(logger_name).critical(message)