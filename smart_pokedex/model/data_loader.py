import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
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
        input_path: Path,
        img_size: tuple[int, int] = (128, 128),
        batch_size: int = 32,
        train_ratio: float = 0.8,
    ) -> None:
        """
        Initializes the Pokémon image data loader.

        Args:
            input_path (Path): Path to the directory containing Pokémon images.
            img_size (tuple[int, int], optional): Target size for resizing images. Defaults to (128, 128).
            batch_size (int, optional): Number of images in each batch. Defaults to 32.
            train_ratio (float, optional): Proportion of data to use for training (between 0 and 1). Defaults to 0.8.

        Raises:
            ValueError: If `train_ratio` is not between 0 and 1.
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1.")

        self._input_path: Path = input_path
        self.img_size: tuple[int, int] = img_size
        self.batch_size: int = batch_size
        self.train_ratio: float = train_ratio

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the image by converting it to the HSV color space, applying edge detection,
        and focusing on color and shape features.

        Args:
            image (np.ndarray): The input image in BGR color space.

        Returns:
            np.ndarray: The preprocessed image in HSV space and with edge detection applied.
        """
        image = tf.image.rgb_to_hsv(image)

        lower_bound = tf.constant([35, 40, 40], dtype=tf.float32)
        upper_bound = tf.constant([85, 255, 255], dtype=tf.float32)

        mask = tf.cast(
            tf.logical_and(image >= lower_bound, image <= upper_bound), tf.float32
        )
        edges = tf.image.sobel_edges(image)
        edges = edges[..., 0]
        combined_features = edges * mask
        combined_resized = tf.image.resize(combined_features, self.img_size)

        return combined_resized / 255.0

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
        _logger.info(f"Loading Pokémon image data from directory: {self._input_path}")
        full_dataset: tf.data.Dataset = (
            tf.keras.preprocessing.image_dataset_from_directory(
                self._input_path,
                image_size=self.img_size,
                batch_size=self.batch_size,
                label_mode="categorical",
                shuffle=True,
                seed=123,
            )
        )
        # Map class indices to class names
        pokemon_class_indices: dict[int, str] = dict(
            enumerate(full_dataset.class_names)
        )

        full_dataset = full_dataset.map(
            lambda x, y: (self._preprocess_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Apply preprocessing (HSV conversion and edge detection) to each image
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * self.train_ratio)
        test_size = dataset_size - train_size

        train_data = full_dataset.take(train_size)
        test_data = full_dataset.skip(train_size).take(test_size)
        _logger.info("Data have been loaded and split into training and test sets.")
        return PokemonImagesData(
            train_data=train_data,
            test_data=test_data,
            pokemon_class_indices=pokemon_class_indices,
        )
