from pathlib import Path

from smart_pokedex.model import METADATA_PATH, MODEL_PATH
from smart_pokedex.model.classifier import PokemonClassifier
from smart_pokedex.model.data_loader import PokemonImagesDataLoader
from smart_pokedex.utils.logger import setup_logging


def run():
    test_path = Path("/home/kruksik/projects/smart_pokedex/train_data")
    val_path = Path("/home/kruksik/projects/smart_pokedex/val_data")
    data = PokemonImagesDataLoader(test_path, val_path).load_data()
    poke_classifier = PokemonClassifier(data=data, epochs=20, img_size=(64, 64))
    poke_classifier.build_model()
    poke_classifier.compile_model()
    poke_classifier.train_model()
    poke_classifier.plot_training_results()
    poke_classifier.evaluate()
    poke_classifier.save_model_with_metadata(
        Path(
            "/home/kruksik/projects/smart_pokedex/smart_pokedex/model/resources/pokemon.keras",
        ),
        Path(
            "/home/kruksik/projects/smart_pokedex/smart_pokedex/model/resources/metadata.json",
        ),
    )


def classification():
    poke_classifier = PokemonClassifier(img_size=(64, 64))
    poke_classifier.load_model_with_metadata(
        MODEL_PATH,
        METADATA_PATH,
    )
    root_dir = Path("/home/kruksik/projects/smart_pokedex/testdata")
    for folder in root_dir.iterdir():
        for _file in folder.iterdir():
            print(poke_classifier.predict(_file), folder.name)


def main() -> None:
    setup_logging()
    # run()
    classification()


if __name__ == "__main__":
    main()
