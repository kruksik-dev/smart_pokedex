from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smart_pokedex.utils.image_loader import PokemonImageLoader


@pytest.fixture
def mock_pokemon_data() -> dict[str, str]:
    return {
        "name": "bulbasaur",
        "sprites": {
            "front_default": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/1.png"
        },
    }


@pytest.mark.parametrize("start_id, end_id", [(1, 2), (1, 3)])
@patch("smart_pokedex.utils.image_loader.requests.get")
def test_download_pokemon_images(
    mock_get: MagicMock,
    mock_output_path: Path,
    start_id: int,
    end_id: int,
    mock_pokemon_data: dict[str, str],
) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = mock_pokemon_data
    mock_response.content = b"fake image content"
    mock_response.raise_for_status.return_value = None
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    loader = PokemonImageLoader()
    loader.download_pokemon_images(mock_output_path, start_id, end_id)

    assert mock_get.call_count == (end_id - start_id + 1) * 2
    for pokemon_id in range(start_id, end_id):
        mock_get.assert_any_call(f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}")


def test_download_images_for_pokemon(
    mock_pokemon_data: dict[str, str],
    mock_output_path: Path,
) -> None:
    with patch(
        "smart_pokedex.utils.image_loader.requests.get", autospec=True
    ) as mock_get, patch(
        "smart_pokedex.utils.image_loader.PokemonImageLoader._save_image_from_url",
        autospec=True,
    ) as mock_save_image:
        loader = PokemonImageLoader()

        loader._download_images_for_pokemon(mock_pokemon_data, mock_output_path)

        mock_save_image.assert_called_once_with(
            mock_output_path / "bulbasaur",
            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/1.png",
            "bulbasaur_1",
        )
        mock_get.assert_not_called()
