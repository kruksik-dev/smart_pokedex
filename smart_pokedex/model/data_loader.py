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

        This method should be overridden in any subclass to define
        specific data loading behavior.

        Returns:
            Any: The data object resulting from the loading process.
        """
        pass


class PokemonImagesDataLoader(Loader):
    """
    Data loader for Pokémon image datasets.

    This class handles loading, preprocessing, and organizing Pokémon image
    datasets for training and validation. It supports resizing images to a
    specified size and batching for TensorFlow processing.

    Attributes:
        _train_data_path (Path): Path to the directory containing training images.
        val_data_path (Path): Path to the directory containing validation images.
        img_size (tuple[int, int]): Dimensions to resize images to (width, height).
        batch_size (int): Number of images per batch during loading.
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
            train_data_path (Path): Path to the directory containing training Pokémon images.
            val_data_path (Path): Path to the directory containing validation Pokémon images.
            img_size (tuple[int, int], optional): Target size for resizing images (width, height). Defaults to (64, 64).
            batch_size (int, optional): Number of images in each batch. Defaults to 32.
        """
        self._train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.img_size: tuple[int, int] = img_size
        self.batch_size: int = batch_size

    def load_data(self) -> PokemonImagesData:
        """
        Loads and preprocesses Pokémon image data.

        This method reads image data from the specified training and validation directories,
        resizes the images to the specified dimensions, normalizes pixel values to the range [0, 1],
        and applies data augmentation (e.g., shear, zoom, horizontal flip) for the training dataset.

        Returns:
            PokemonImagesData: An object containing:
                - `train_data` (tf.keras.preprocessing.image.DirectoryIterator):
                  TensorFlow Dataset object for training data.
                - `test_data` (tf.keras.preprocessing.image.DirectoryIterator):
                  TensorFlow Dataset object for validation data.
                - `pokemon_class_indices` (dict): Mapping of class indices to class names.

        Raises:
            FileNotFoundError: If the specified training or validation directory does not exist.
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
            batch_size=self.batch_size,
            class_mode="categorical",
            color_mode="rgb",
        )
        validation_set = datagen.flow_from_directory(
            self.val_data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
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
