import logging
from io import BytesIO
from pathlib import Path
from typing import Generator, Optional

import requests
from PIL import Image

_logger = logging.getLogger(__name__)


class PokemonImageLoader:
    """
    A class to download Pokémon images from the PokeAPI and save them locally.

    Attributes:
        _POKEMON_API_URL (str): The base URL for the PokeAPI to fetch Pokémon data.
    """

    _POKEMON_API_URL = "https://pokeapi.co/api/v2/pokemon"

    def download_pokemon_images(
        self, output_path: Path, start_id: int = 1, end_id: int = 152
    ) -> None:
        """
        Downloads images of Pokémon in the given ID range and saves them to the specified directory.

        Args:
            output_path (Path): The path where Pokémon images will be saved.
            start_id (int): The ID of the first Pokémon to download (inclusive). Defaults to 1.
            end_id (int): The ID of the last Pokémon to download (exclusive). Defaults to 152.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        for pokemon_id in range(start_id, end_id):
            url = f"{self._POKEMON_API_URL}/{pokemon_id}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                pokemon_data = response.json()
                self._download_images_for_pokemon(pokemon_data, output_path)
            except requests.exceptions.RequestException as e:
                _logger.warning(
                    f"Failed to fetch data for Pokémon ID {pokemon_id}: {e}"
                )
            except ValueError as e:
                _logger.warning(
                    f"Failed to parse JSON for Pokémon ID {pokemon_id}: {e}"
                )

    def _download_images_for_pokemon(
        self, pokemon_data: dict, output_path: Path
    ) -> None:
        """
        Helper function to download all images for a specific Pokémon and save them.

        Args:
            pokemon_data (dict): The Pokémon data containing sprite URLs.
            output_path (Path): The base path where images will be saved.
        """
        pokemon_name = pokemon_data.get("name")
        if not pokemon_name:
            _logger.warning(f"Missing name for Pokémon data: {pokemon_data}")
            return

        pokemon_output_path = output_path / pokemon_name
        pokemon_output_path.mkdir(exist_ok=True)

        image_id = 1
        for url in self._get_pokemon_image_urls(pokemon_data["sprites"]):
            image_name = f"{pokemon_name}_{image_id}"
            self._save_image_from_url(pokemon_output_path, url, image_name)
            image_id += 1

    @staticmethod
    def _get_pokemon_image_urls(
        data: dict, seen_urls: Optional[set] = None
    ) -> Generator[str, None, None]:
        """
        Extracts all unique image URLs from a nested dictionary of sprite data.

        Args:
            data (dict): The nested dictionary containing Pokémon sprite URLs.
            seen_urls (set, optional): A set to track already seen URLs. Defaults to None.

        Yields:
            str: A unique image URL.
        """
        if seen_urls is None:
            seen_urls = set()

        for value in data.values():
            if isinstance(value, dict):
                yield from PokemonImageLoader._get_pokemon_image_urls(value, seen_urls)
            elif (
                isinstance(value, str)
                and value.startswith("https://")
                and value.endswith(".png")
            ):
                if value not in seen_urls:
                    seen_urls.add(value)
                    yield value

    @staticmethod
    def _save_image_from_url(save_path: Path, image_url: str, name: str) -> None:
        """
        Downloads an image from a URL and saves it to the specified path.

        Args:
            save_path (Path): The directory where the image will be saved.
            image_url (str): The URL of the image.
            name (str): The name to use when saving the image.
        """
        try:
            img_response = requests.get(image_url)
            img_response.raise_for_status()

            img = Image.open(BytesIO(img_response.content))
            img = img.resize((128, 128))
            img_path = save_path / f"{name}.png"
            img.save(img_path)
            _logger.info(f"Downloaded and saved image '{name}' under {img_path}")
        except requests.exceptions.RequestException as e:
            _logger.warning(f"Failed to download image from {image_url}: {e}")
        except IOError as e:
            _logger.warning(f"Failed to save image {name} from {image_url}: {e}")
