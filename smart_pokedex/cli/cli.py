from pathlib import Path

from smart_pokedex.model import METADATA_PATH, MODEL_PATH
from smart_pokedex.model.classifier import PokemonClassifier
from smart_pokedex.model.data_loader import PokemonImagesDataLoader
from smart_pokedex.utils.logger import setup_logging


def run():
    path_to_resources = Path("/home/kruksik/own_projects/smart_pokedex/resources_new")
    data = PokemonImagesDataLoader(path_to_resources).load_data()
    poke_classifier = PokemonClassifier(data=data, epochs=100)
    poke_classifier.build_model()
    poke_classifier.compile_model()
    poke_classifier.train_model()
    poke_classifier.evaluate()
    poke_classifier.save_model_with_metadata(
        Path(
            "/home/kruksik/own_projects/smart_pokedex/pokemon.keras",
            "/home/kruksik/own_projects/smart_pokedex/metadata.json",
        )
    )


def classification():
    poke_classifier = PokemonClassifier()
    poke_classifier.load_model_with_metadata(
        MODEL_PATH,
        METADATA_PATH,
    )
    root_dir = Path("/home/kruksik/own_projects/smart_pokedex/resources_new/")
    for folder in root_dir.iterdir():
        for _file in folder.iterdir():
            print(poke_classifier.predict(_file))


def main() -> None:
    setup_logging()
    # run()
    classification()


if __name__ == "__main__":
    main()
