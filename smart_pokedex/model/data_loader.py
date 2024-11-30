import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import tensorflow as tf

from smart_pokedex.model.image_data import PokemonImagesData

_logger = logging.getLogger(__name__)


class Loader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load_data(self) -> Any:
        """
        Load data method to be implemented by subclasses.
        Must be implemented by subclasses.

        Returns:
            Any: Loaded data object.
        """
        pass


class PokemonImagesDataLoader(Loader):
    """
    Data loader for Pokémon image datasets.

    This loader is designed to handle loading, preprocessing, and splitting
    Pokémon image datasets for training and testing.

    Attributes:
        _input_path (Path): Path to the directory containing Pokémon images.
        img_size (tuple[int, int]): Size to which images will be resized.
        batch_size (int): Number of images in each batch.
        train_ratio (float): Proportion of data to use for training (between 0 and 1).
    """

    def __init__(
        self,
        train_data_path: Path,
        val_data_path: Path,
        img_size: tuple[int, int] = (64, 64),
        batch_size: int = 32,
    ) -> None:
        """
        Initializes the Pokémon image data loader.

        Args:
            input_path (Path): Path to the directory containing Pokémon images.
            img_size (tuple[int, int], optional): Target size for resizing images. Defaults to (128, 128).
            batch_size (int, optional): Number of images in each batch. Defaults to 32.
            train_ratio (float, optional): Proportion of data to use for training (between 0 and 1). Defaults to 0.8.
        """

        self._train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.img_size: tuple[int, int] = img_size
        self.batch_size: int = batch_size

    def load_data(self) -> PokemonImagesData:
        """
        Loads and preprocesses Pokémon image data.

        The method loads image data from the specified directory, resizes the images,
        normalizes pixel values to the range [0, 1], and splits the dataset into
        training and test subsets based on the provided training ratio.

        Returns:
            PokemonImagesData: A data object containing:
                - `train_data`: Training dataset as a TensorFlow `Dataset` object.
                - `test_data`: Test dataset as a TensorFlow `Dataset` object.
                - `pokemon_class_indices`: Mapping of class indices to class names.

        Raises:
            FileNotFoundError: If the input directory does not exist.
        """
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )
        training_set = datagen.flow_from_directory(
            self._train_data_path,
            target_size=self.img_size,
            batch_size=32,
            class_mode="categorical",
            color_mode="rgb",
        )
        validation_set = datagen.flow_from_directory(
            self.val_data_path,
            target_size=self.img_size,
            batch_size=32,
            class_mode="categorical",
            color_mode="rgb",
        )
        pokemon_class_indices = {
            value: key for key, value in training_set.class_indices.items()
        }
        return PokemonImagesData(
            train_data=training_set,
            test_data=validation_set,
            pokemon_class_indices=pokemon_class_indices,
        )
