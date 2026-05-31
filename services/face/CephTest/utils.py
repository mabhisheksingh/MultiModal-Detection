import yaml
import logging
import os
from typing import Dict, Any


class ConfigLoader:
    """Handles loading and validation of configuration from YAML file."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the ConfigLoader.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = None
        
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration data
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If required configuration is missing
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create a config.yaml file with your Ceph credentials."
            )
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")
        
        self._validate_config()
        return self.config
    
    def _validate_config(self):
        """
        Validate that required configuration fields are present.
        
        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not self.config:
            raise ValueError("Configuration is empty")
        
        required_fields = {
            'auth': ['access_key', 'secret_key'],
            'endpoint': ['url']
        }
        
        for section, fields in required_fields.items():
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
            
            for field in fields:
                if field not in self.config[section]:
                    raise ValueError(
                        f"Missing required field: {section}.{field}"
                    )
                
                value = self.config[section][field]
                if not value or (isinstance(value, str) and 
                               ('YOUR_' in value or 'your-' in value)):
                    raise ValueError(
                        f"Please configure {section}.{field} in {self.config_path}"
                    )
    
    def get(self, *keys, default=None):
        """
        Get a configuration value using dot notation.
        
        Args:
            *keys: Keys to traverse the configuration dictionary
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self.config:
            self.load()
        
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class LoggerSetup:
    """Handles logging configuration and setup."""
    
    @staticmethod
    def setup_logger(config: Dict[str, Any]) -> logging.Logger:
        """
        Set up logging based on configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured logger instance
        """
        log_config = config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('log_file', 'ceph_operations.log')
        console_output = log_config.get('console_output', True)
        
        logger = logging.getLogger('CephClient')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def validate_file_path(file_path: str) -> str:
    """
    Validate that a file path exists and is accessible.
    
    Args:
        file_path: Path to file
        
    Returns:
        Absolute path to file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"File is not readable: {file_path}")
    
    return os.path.abspath(file_path)
