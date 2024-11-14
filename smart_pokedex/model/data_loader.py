import logging
from pathlib import Path
from typing import Any
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from abc import ABC, abstractmethod

_logger = logging.getLogger(__name__)


class Loader(ABC):
    @abstractmethod
    def load_data(self) -> Any:
        pass


class PokemonImagesDataLoader(Loader):
    def __init__(
        self,
        input_path: Path,
        img_size: tuple[int, int] = (128, 128),
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._input_path = input_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.load_data()

    def load_data(self) -> None:
        _logger.info("Starting loading input data ...")
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            fill_mode="nearest",
        )
        self.train_data: DirectoryIterator = datagen.flow_from_directory(
            self._input_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
        )

        self.val_data: DirectoryIterator = datagen.flow_from_directory(
            self._input_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
        )
        self.num_classes = len(self.train_data.class_indices)
        _logger.info("Data have been loaded")
