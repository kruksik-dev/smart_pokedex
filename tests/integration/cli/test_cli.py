from unittest.mock import patch

import pytest
from click.testing import CliRunner

from smart_pokedex.cli.cli import run_pokemon_classification
from tests.integration.conftest import IMAGES_RESOURCES_PATH


@pytest.mark.parametrize(
    "image_name, expected_prediction_name",
    (
        pytest.param("test_case_1.jpg", "Abra", id="test_case_1"),
        pytest.param("test_case_2.jpg", "Abra", id="test_case_2"),
        pytest.param("test_case_3.jpg", "Electrode", id="test_case_3"),
        pytest.param("test_case_4.jpg", "Electrode", id="test_case_4"),
    ),
)
def test_cli_run(image_name: str, expected_prediction_name: str) -> None:
    with patch("smart_pokedex.cli.cli.display_prediction") as mock_display:
        runner = CliRunner()
        result = runner.invoke(
            run_pokemon_classification,
            ["-i", str(IMAGES_RESOURCES_PATH / image_name)],
        )

        assert result.exit_code == 0
        mock_display.assert_called_once()
        assert mock_display.call_args[0][0] == expected_prediction_name
