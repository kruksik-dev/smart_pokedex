import os
from pathlib import Path

import click
import pyfiglet
from colorama import Fore, Style

from smart_pokedex.model import METADATA_PATH, MODEL_PATH
from smart_pokedex.model.classifier import PokemonClassifier
from smart_pokedex.utils.logger import setup_logging


def display_prediction(predicted_class: str, confidence: float) -> None:
    """
    Display a styled prediction result in the terminal.

    Args:
        predicted_class (str): The predicted Pokémon species.
        confidence (float): Confidence percentage of the prediction.
    """

    ascii_art = pyfiglet.figlet_format(predicted_class.upper())

    print(Fore.YELLOW + "=" * 50)
    print(Fore.CYAN + ascii_art)
    print(Fore.GREEN + f"Confidence Level: {confidence:.2f}%")
    print(Fore.YELLOW + "=" * 50 + Style.RESET_ALL)


@click.command()
@click.option(
    "-i",
    "--image_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to pokemon image file",
)
def run_pokemon_classification(image_path: Path) -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    setup_logging()
    poke_classifier = PokemonClassifier(img_size=(64, 64))
    poke_classifier.load_model_with_metadata(
        MODEL_PATH,
        METADATA_PATH,
    )
    predicted_class, confidence = poke_classifier.predict(image_path)
    display_prediction(predicted_class, confidence)


if __name__ == "__main__":
    run_pokemon_classification()
