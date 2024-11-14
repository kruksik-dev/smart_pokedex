from pathlib import Path
from smart_pokedex.model.classifier import PokemonClassifier
from smart_pokedex.model.data_loader import PokemonImagesDataLoader
from smart_pokedex.utils.logger import setup_logging


def main() -> None:
    setup_logging()
    path_to_resources = Path("/workspaces/smart_pokedex/resources")
    data_loader = PokemonImagesDataLoader(path_to_resources)
    poke_classifier = PokemonClassifier(data_loader, epochs=40)
    for _ in range(3):
        poke_classifier.load_model(Path("best_model.keras"))
        poke_classifier.train_model()
        poke_classifier.evaluate()


if __name__ == "__main__":
    main()
