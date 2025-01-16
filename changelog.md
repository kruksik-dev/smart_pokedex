# Changelog

## [0.0.2]

### Added
- Dockerization has been added, which allows you to run the application inside a container
- The prepared image was placed directly on the dockerhub: kruksik/smart pokedex
- Updated CICD 

## [0.0.1]

### Added
- Added CLI command for Pokémon classification in [`smart_pokedex/cli/cli.py`](smart_pokedex/cli/cli.py).
- Added `PokemonClassifier` class for building, training, evaluating, saving, and loading machine learning models in [`smart_pokedex/model/classifier.py`](smart_pokedex/model/classifier.py).
- Added `PokemonImagesDataLoader` class for loading and preprocessing Pokémon image datasets in [`smart_pokedex/model/data_loader.py`](smart_pokedex/model/data_loader.py).
- Added `PokemonImagesData` dataclass for storing Pokémon image data in [`smart_pokedex/model/image_objects.py`](smart_pokedex/model/image_objects.py).
- Added `metadata.json` file containing Pokémon class indices in [`smart_pokedex/model/resources/metadata.json`](smart_pokedex/model/resources/metadata.json).
- Added `pokemon.keras` file for storing the trained model in [`smart_pokedex/model/resources/pokemon.keras`](smart_pokedex/model/resources/pokemon.keras).
- Added `PokemonImageLoader` class for downloading Pokémon images from the PokeAPI in [`smart_pokedex/utils/image_loader.py`](smart_pokedex/utils/image_loader.py).
- Added logging setup functions in [`smart_pokedex/utils/logger.py`](smart_pokedex/utils/logger.py).
- Added CICD actions with testing stage
