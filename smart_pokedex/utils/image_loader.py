import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Generator

import requests
from PIL import Image

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class PokemonImageLoader:
    _POKEMON_API_URL = "https://pokeapi.co/api/v2/pokemon"

    def download_pokemon_images(
        self, output_path: Path, start_id: int = 1, end_id: int = 152
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        for pokemon_id in range(start_id, end_id):
            url = f"{self._POKEMON_API_URL}/{pokemon_id}"
            response = requests.get(url)
            if response.status_code == 200:
                image_id = 1
                pokemon_data = response.json()
                pokemon_output_path = output_path / pokemon_data["name"]
                for url in self._get_pokemon_image_url(pokemon_data["sprites"]):
                    image_name = f"{pokemon_data['name']}_{image_id}"
                    image_id += 1
                    self._save_image_from_url(pokemon_output_path, url, image_name)

    @staticmethod
    def _get_pokemon_image_url(
        data: dict, seen_urls: set = None
    ) -> Generator[str, Any, Any]:
        if seen_urls is None:
            seen_urls = set()
        for _, value in data.items():
            if isinstance(value, dict):
                yield from PokemonImageLoader._get_pokemon_image_url(value, seen_urls)
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
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            save_path.mkdir(exist_ok=True)
            img = Image.open(BytesIO(img_response.content))
            img = img.resize((128, 128))
            img_path = save_path / f"{name}.png"
            img.save(img_path)
            _logger.info(f"Downloaded and saved {name} image under {img_path}")


PokemonImageLoader().download_pokemon_images(
    Path("/home/kruksik/own_projects/smart_pokedex")
)
