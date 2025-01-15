import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from pytest import MonkeyPatch

from smart_pokedex import ROOT_PATH
from smart_pokedex.utils.logger import get_log_config_path, setup_logging


def test_get_log_config_path_env_var_set(monkeypatch: MonkeyPatch) -> None:
    expected_path = "/path/to/logging.yml"
    monkeypatch.setenv("LOGGING_CONFIG_PATH", expected_path)

    result = get_log_config_path()

    assert result == Path(expected_path)


def test_get_log_config_path_env_var_not_set(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("LOGGING_CONFIG_PATH", raising=False)
    expected_path = ROOT_PATH / "logging.yml"

    result = get_log_config_path()

    assert result == expected_path


@patch("smart_pokedex.utils.logger.get_log_config_path")
@patch("smart_pokedex.utils.logger.logging.config.dictConfig")
def test_setup_logging_with_config_file(
    mock_dictConfig: MagicMock, mock_get_log_config_path: MagicMock, tmp_path: Path
) -> None:
    log_config_path = tmp_path / "logging.yml"
    log_config_content = {
        "version": 1,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
            }
        },
        "root": {
            "handlers": ["console"],
            "level": "DEBUG",
        },
    }

    with open(log_config_path, "w") as f:
        yaml.dump(log_config_content, f)

    mock_get_log_config_path.return_value = log_config_path

    setup_logging()

    mock_dictConfig.assert_called_once_with(log_config_content)


@patch("smart_pokedex.utils.logger.get_log_config_path")
@patch("smart_pokedex.utils.logger.logging.basicConfig")
def test_setup_logging_without_config_file(
    mock_basicConfig: MagicMock, mock_get_log_config_path: MagicMock, tmp_path: Path
) -> None:
    log_config_path = tmp_path / "logging.yml"
    mock_get_log_config_path.return_value = log_config_path

    setup_logging()

    mock_basicConfig.assert_called_once_with(level=logging.INFO)
