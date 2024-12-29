from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smart_pokedex.utils.image_loader import PokemonImageLoader


@pytest.fixture
def mock_output_path(tmp_path: Path) -> Path:
    return tmp_path / "pokemon_images"


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
    mock_get, mock_output_path, start_id, end_id, mock_pokemon_data
):
    mock_response = MagicMock()
    mock_response.json.return_value = mock_pokemon_data
    mock_response.content = b"fake image content"
    mock_response.raise_for_status.return_value = None
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    loader = PokemonImageLoader()
    loader.download_pokemon_images(mock_output_path, start_id, end_id)

    assert mock_get.call_count == (end_id - start_id)
    for pokemon_id in range(start_id, end_id):
        mock_get.assert_any_call(f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}")


@patch("smart_pokedex.utils.image_loader.requests.get")
@patch("smart_pokedex.utils.image_loader.PokemonImageLoader._save_image_from_url")
def test_download_images_for_pokemon(
    mock_save_image: MagicMock,
    mock_output_path: MagicMock,
    mock_pokemon_data: MagicMock,
) -> None:
    loader = PokemonImageLoader()
    loader._download_images_for_pokemon(mock_pokemon_data, mock_output_path)

    mock_save_image.assert_called_once_with(
        mock_output_path / "bulbasaur",
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/1.png",
        "bulbasaur_1",
    )


def test_get_pokemon_image_urls(mock_pokemon_data):
    urls = list(
        PokemonImageLoader._get_pokemon_image_urls(mock_pokemon_data["sprites"])
    )
    assert urls == [
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/1.png"
    ]


@patch("smart_pokedex.utils.image_loader.requests.get")
@patch("smart_pokedex.utils.image_loader.Image.open")
def test_save_image_from_url(
    mock_image_open: MagicMock, mock_requests_get: MagicMock
) -> None:

    mock_response = MagicMock()
    mock_response.content = b"fake_image_data"
    mock_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_response

    mock_image = MagicMock()
    mock_resized_image = MagicMock()
    mock_image.resize.return_value = mock_resized_image
    mock_image_open.return_value = mock_image

    save_path = Path("/tmp/test_pokemon_images")
    mock_save_path = MagicMock()
    mock_save_path.__truediv__.return_value = save_path / "test_image.png"

    PokemonImageLoader._save_image_from_url(
        save_path, "https://example.com/image.png", "test_image"
    )

    mock_requests_get.assert_called_once_with("https://example.com/image.png")
    mock_image.resize.assert_called_once_with((128, 128))
    mock_resized_image.save.assert_called_once_with(save_path / "test_image.png")
