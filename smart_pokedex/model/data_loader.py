import logging
from pathlib import Path
from typing import Any

import tensorflow as tf

_logger = logging.getLogger(__name__)


class Loader:
    def load_data(self) -> Any:
        pass


class PokemonImagesDataLoader(Loader):
    def __init__(
        self,
        input_path: Path,
        img_size: tuple[int, int] = (128, 128),
        batch_size: int = 32,
        train_ratio: float = 0.8,
    ) -> None:
        super().__init__()
        self._input_path = input_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.load_data()

    def load_data(self) -> None:
        _logger.info("Starting loading input data ...")

        # Wczytujemy dane z katalogu
        full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self._input_path,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=True,
            seed=123,
        )
        self.pokemon_class_indices = {
            i: class_name for i, class_name in enumerate(full_dataset.class_names)
        }
        full_dataset = full_dataset.map(lambda x, y: (x / 255, y))

        train_size = int(len(full_dataset) * self.train_ratio)
        test_size = int(len(full_dataset) * 1 - self.train_ratio)

        self.train_data = full_dataset.take(train_size)
        self.test_data = full_dataset.skip(train_size).take(test_size)

        _logger.info("Data have been loaded and split into training and test sets.")
