import logging
import logging.config
import os
from pathlib import Path

import yaml


def get_log_config_path() -> Path:
    """
    Retrieves the path to the logging configuration file.

    First, checks if an environment variable `LOGGING_CONFIG_PATH` is set.
    If set, it returns that path. If not, it defaults to looking for
    `logging.yml` in the same directory as the current script.

    Returns:
        Path: The path to the logging configuration file.
    """
    log_config_path = os.getenv("LOGGING_CONFIG_PATH")
    if log_config_path:
        return Path(log_config_path)
    return Path(__file__).parent.resolve() / "logging.yml"


def setup_logging() -> None:
    """
    Sets up logging configuration.

    This function loads logging configuration from a YAML file. The file
    path is determined by the `get_log_config_path()` function. If the YAML
    configuration file is found, it is loaded using `logging.config.dictConfig()`.
    If the file is not found, a basic logging configuration with the INFO level
    is set up.

    If the logging configuration is successfully loaded, it configures the logging
    behavior for the application; otherwise, a default logging setup is applied.

    Returns:
        None
    """
    log_config_path: Path = get_log_config_path()

    if log_config_path.exists():
        with open(log_config_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
