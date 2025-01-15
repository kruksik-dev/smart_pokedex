from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from smart_pokedex.utils.image_loader import PokemonImageLoader


@pytest.mark.parametrize(
    "mocked_sprites_data, expected_urls",
    [
        pytest.param(
            {"front_default": "https://example.com/image.png"},
            ["https://example.com/image.png"],
            id="https_valid_url",
        ),
        pytest.param({"front_default": "invalid_url"}, [], id="invalid_url"),
        pytest.param(
            {"front_default": "http://example.com/image.png"},
            [],
            id="invalid_url_no_secured",
        ),
    ],
)
def test_get_pokemon_image_urls(
    mocked_sprites_data: dict[str, str], expected_urls: list[str]
) -> None:
    urls = list(PokemonImageLoader._get_pokemon_image_urls(mocked_sprites_data))
    assert urls == expected_urls


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


@patch("smart_pokedex.utils.image_loader.requests.get")
def test_download_pokemon_images_request_exception(
    mock_get: MagicMock, mock_output_path: Path
) -> None:
    mock_get.side_effect = requests.exceptions.RequestException("Request failed")

    loader = PokemonImageLoader()
    loader.download_pokemon_images(mock_output_path, start_id=1, end_id=2)

    assert mock_get.call_count == 2


@patch("smart_pokedex.utils.image_loader.requests.get")
def test_download_pokemon_images_value_error(
    mock_get: MagicMock, mock_output_path: Path
) -> None:
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_get.return_value = mock_response

    loader = PokemonImageLoader()
    loader.download_pokemon_images(mock_output_path, start_id=1, end_id=2)

    assert mock_get.call_count == 2


def test_missing_pokemon_name_logs_warning() -> None:

    with patch("smart_pokedex.utils.image_loader.requests.get") as mock_get, patch(
        "smart_pokedex.utils.image_loader._logger"
    ) as mock_logger:
        mock_get.return_value.json.return_value = {
            "sprites": {
                "front_default": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/1.png"
            }
        }
        mock_pokemon_data = mock_get.return_value.json.return_value
        mock_output_path = Path("/tmp")
        loader = PokemonImageLoader()

        loader._download_images_for_pokemon(mock_pokemon_data, mock_output_path)

        mock_logger.warning.assert_called_once_with(
            f"Missing name for Pokémon data: {mock_pokemon_data}"
        )


def test_get_pokemon_image_urls_with_mocked_urls() -> None:
    mock_data = {
        "front_default": "https://mockurl.com/pokemon/1.png",
        "other": {
            "official-artwork": {
                "front_default": "https://mockurl.com/pokemon/other/official-artwork/1.png"
            },
            "home": {"front_default": "https://mockurl.com/pokemon/other/home/1.png"},
        },
    }
    expected_urls = {
        "https://mockurl.com/pokemon/1.png",
        "https://mockurl.com/pokemon/other/official-artwork/1.png",
        "https://mockurl.com/pokemon/other/home/1.png",
    }
    seen_urls = set()

    with patch.object(
        PokemonImageLoader,
        "_get_pokemon_image_urls",
        wraps=PokemonImageLoader._get_pokemon_image_urls,
    ) as mock_get_pokemon_image_urls:
        result = set(PokemonImageLoader._get_pokemon_image_urls(mock_data, seen_urls))

        assert result == expected_urls
        assert mock_get_pokemon_image_urls.call_count > 1


def test_save_image_from_url_request_exception() -> None:
    save_path = "/mock/path"
    image_url = "https://mockurl.com/image.png"
    name = "test_image"

    with patch(
        "requests.get",
        side_effect=requests.exceptions.RequestException("Network Error"),
    ) as mock_get, patch("smart_pokedex.utils.image_loader._logger") as mock_logger:
        loader = PokemonImageLoader()

        loader._save_image_from_url(save_path, image_url, name)

        mock_get.assert_called_once_with(image_url)
        mock_logger.warning.assert_called_once_with(
            f"Failed to download image from {image_url}: Network Error"
        )
