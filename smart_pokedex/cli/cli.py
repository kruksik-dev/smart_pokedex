import os
from pathlib import Path

from smart_pokedex.model.classifier import PokemonClassifier
from smart_pokedex.model.data_loader import PokemonImagesDataLoader
from smart_pokedex.model.predictor import PokemonPredictor
from smart_pokedex.utils.logger import setup_logging


def run():
    path_to_resources = Path("/home/kruksik/own_projects/smart_pokedex/resources_new")
    data_loader = PokemonImagesDataLoader(path_to_resources)
    poke_classifier = PokemonClassifier(data_loader, epochs=50)
    poke_classifier.build_model()
    poke_classifier.compile_model()
    poke_classifier.train_model()
    poke_classifier.evaluate()
    poke_classifier.save_model(
        Path("/home/kruksik/own_projects/smart_pokedex/pokemon.keras")
    )


def classification():
    path_to_resources = Path("/home/kruksik/own_projects/smart_pokedex/resources_new")
    data_loader = PokemonImagesDataLoader(path_to_resources)
    poke_classifier = PokemonClassifier()
    poke_classifier.load_model(
        Path(
            "/home/kruksik/own_projects/smart_pokedex/smart_pokedex/cli/best_model.keras"
        )
    )
    predictor = PokemonPredictor(
        poke_classifier.model, data_loader.pokemon_class_indices
    )
    for root, dirs, files in os.walk(
        "/home/kruksik/own_projects/smart_pokedex/resources"
    ):
        poke_name = os.path.basename(root)
        counter = 0
        for img in files:
            file_path = os.path.join(root, img)
            pokemon_name = predictor.predict(file_path)
            print(f"For img {img} pokemon {poke_name} name is: {pokemon_name}")
            if poke_name == pokemon_name:
                counter += 1
        print(f"Accuracy for {poke_name}: {counter}/{len(files)}")


def main() -> None:
    setup_logging()
    # run()
    classification()


if __name__ == "__main__":
    main()
