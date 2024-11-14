import logging
import logging.config
import os
import yaml
from pathlib import Path


def get_log_config_path() -> Path:
    log_config_path = os.getenv("LOGGING_CONFIG_PATH")
    if log_config_path:
        return Path(log_config_path)
    return Path(__file__).parent.resolve() / "logging.yml"


def setup_logging() -> None:
    log_config_path: Path = get_log_config_path()

    if log_config_path.exists():
        with open(log_config_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
