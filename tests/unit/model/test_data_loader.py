from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from smart_pokedex.model.data_loader import Loader, PokemonImagesDataLoader
from smart_pokedex.model.image_objects import PokemonImagesData


class MockLoader(Loader):
    def load_data(self) -> Any:
        return {"mock": "data"}


@pytest.fixture
def data_loader() -> PokemonImagesDataLoader:
    train_data_path = Path("/path/to/train_data")
    val_data_path = Path("/path/to/val_data")
    img_size = (64, 64)
    batch_size = 32
    return PokemonImagesDataLoader(train_data_path, val_data_path, img_size, batch_size)


def test_loader_abstract_class() -> None:
    with pytest.raises(TypeError):
        Loader()


def test_mock_loader() -> None:
    loader = MockLoader()
    result = loader.load_data()
    assert result == {"mock": "data"}, f"Expected {{'mock': 'data'}} but got {result}"


@patch(
    "smart_pokedex.model.data_loader.tf.keras.preprocessing.image.ImageDataGenerator"
)
def test_load_data(
    mock_image_data_generator: MagicMock,
    data_loader: PokemonImagesDataLoader,
) -> None:
    mock_datagen = MagicMock()
    mock_image_data_generator.return_value = mock_datagen

    mock_training_set = MagicMock()
    mock_validation_set = MagicMock()
    mock_datagen.flow_from_directory.side_effect = [
        mock_training_set,
        mock_validation_set,
    ]

    mock_training_set.class_indices = {"class1": 0, "class2": 1}

    result = data_loader.load_data()
    img_size = (64, 64)
    batch_size = 32

    mock_image_data_generator.assert_called_once_with(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    mock_datagen.flow_from_directory.assert_any_call(
        Path("/path/to/train_data"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
    )
    mock_datagen.flow_from_directory.assert_any_call(
        Path("/path/to/val_data"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
    )

    assert isinstance(result, PokemonImagesData)
    assert result.train_data == mock_training_set
    assert result.test_data == mock_validation_set
    assert result.pokemon_class_indices == {0: "class1", 1: "class2"}
