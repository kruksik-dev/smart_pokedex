import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

from smart_pokedex.model.classifier import PokemonClassifier
from smart_pokedex.model.data_loader import PokemonImagesData


@pytest.fixture
def mock_data() -> MagicMock:
    mock_data = MagicMock(spec=PokemonImagesData)
    mock_data.train_data = MagicMock()
    mock_data.test_data = MagicMock()
    mock_data.pokemon_class_indices = {"0": "Pikachu", "1": "Bulbasaur"}
    return mock_data


@pytest.fixture
def pokemon_classifier(mock_data: MagicMock) -> PokemonClassifier:
    return PokemonClassifier(data=mock_data)


def test_build_model(pokemon_classifier: PokemonClassifier) -> None:
    pokemon_classifier.build_model()
    assert pokemon_classifier.model is not None
    assert isinstance(pokemon_classifier.model, tf.keras.Model)


def test_compile_model(pokemon_classifier: PokemonClassifier) -> None:
    pokemon_classifier.build_model()
    pokemon_classifier.compile_model()
    assert pokemon_classifier.model.optimizer is not None
    assert pokemon_classifier.model.loss == "categorical_crossentropy"


def test_train_model(pokemon_classifier: PokemonClassifier) -> None:
    pokemon_classifier.build_model()
    pokemon_classifier.compile_model()

    pokemon_classifier.model.fit = MagicMock(return_value="training complete")
    pokemon_classifier.train_model()
    pokemon_classifier.model.fit.assert_called_once()


def test_evaluate_model(pokemon_classifier: PokemonClassifier) -> None:
    pokemon_classifier.build_model()
    pokemon_classifier.compile_model()

    pokemon_classifier.model.evaluate = MagicMock(return_value=(0.0, 0.95))
    accuracy = pokemon_classifier.evaluate()
    assert accuracy == 0.95


def test_save_model(pokemon_classifier: PokemonClassifier) -> None:
    with tempfile.TemporaryDirectory() as tmp_path, patch(
        "tensorflow.keras.Model.save"
    ) as mock_save:
        pokemon_classifier.build_model()
        path = Path(tmp_path)
        pokemon_classifier.save_model(path)
        mock_save.assert_called_once_with(path)


def test_load_model(pokemon_classifier: PokemonClassifier) -> None:
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmp_file, patch(
        "tensorflow.keras.models.load_model", return_value=MagicMock()
    ) as mock_load:
        path = Path(tmp_file.name)
        pokemon_classifier.load_model(path)
        mock_load.assert_called_once_with(path)


def test_save_model_with_metadata(pokemon_classifier: PokemonClassifier) -> None:
    with tempfile.TemporaryDirectory() as tmp_path, patch(
        "builtins.open", new_callable=MagicMock
    ) as mock_open, patch("tensorflow.keras.Model.save") as mock_save:
        model_path = Path(tmp_path) / "model.keras"
        metadata_path = Path(tmp_path) / "metadata.json"
        pokemon_classifier.build_model()
        pokemon_classifier.save_model_with_metadata(model_path, metadata_path)
        mock_save.assert_called_once_with(model_path)
        mock_open.assert_called_once_with(metadata_path, "w")


def test_load_model_with_metadata(pokemon_classifier: PokemonClassifier) -> None:
    with patch(
        "tensorflow.keras.models.load_model", return_value=MagicMock()
    ) as mock_load, patch(
        "builtins.open", new_callable=MagicMock
    ) as mock_open, tempfile.NamedTemporaryFile(
        suffix=".keras", delete=False
    ) as tmp_keras_file, tempfile.NamedTemporaryFile(
        suffix=".json", delete=False
    ) as tmp_json_file:
        mock_open.return_value.__enter__.return_value.read.return_value = (
            '{"0": "Pikachu", "1": "Bulbasaur"}'
        )

        model_path = Path(tmp_keras_file.name)
        metadata_path = Path(tmp_json_file.name)
        pokemon_classifier.load_model_with_metadata(model_path, metadata_path)

        mock_load.assert_called_once_with(model_path)
        mock_open.assert_called_once_with(metadata_path, "r")
        assert pokemon_classifier.pokemon_class_indices == {
            "0": "Pikachu",
            "1": "Bulbasaur",
        }


def test_predict(pokemon_classifier: PokemonClassifier) -> None:
    pokemon_classifier.build_model()
    pokemon_classifier.compile_model()

    with patch("tensorflow.keras.preprocessing.image.load_img") as mock_load_img, patch(
        "tensorflow.keras.preprocessing.image.img_to_array"
    ) as mock_img_to_array, patch(
        "numpy.expand_dims", side_effect=lambda x, axis: np.array([x])
    ):

        mock_load_img.return_value = MagicMock()
        mock_img_to_array.return_value = np.random.rand(128, 128, 3)
        pokemon_classifier.model.predict = MagicMock(
            return_value=np.array([[0.1, 0.9]])
        )

        image_path = Path("/fake/path/image.png")
        predicted_class, confidence = pokemon_classifier.predict(image_path)

        assert predicted_class == "Bulbasaur"
        assert pytest.approx(confidence, 0.1) == 90.0
