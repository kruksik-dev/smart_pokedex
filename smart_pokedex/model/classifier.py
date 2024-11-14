import logging
from pathlib import Path
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from smart_pokedex.model.data_loader import PokemonImagesDataLoader
from tensorflow.keras.layers import BatchNormalization

_logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self) -> None:
        self.model = None

    @abstractmethod
    def build_model(self) -> None:
        pass

    @abstractmethod
    def compile_model(self) -> None:
        pass

    @abstractmethod
    def train_model(self) -> None:
        pass


class PokemonClassifier(BaseModel):
    def __init__(self, data_loader: PokemonImagesDataLoader, epochs: int = 20) -> None:
        super().__init__()
        self.epochs = epochs
        self.data_loader = data_loader
        self.training_results = None
        self.data_loader_validation()

    def data_loader_validation(self) -> None:
        _logger.debug("Starting data loader validation ...")
        required_attributes = ["train_data", "val_data", "num_classes"]
        missing_attributes = [
            attr for attr in required_attributes if not hasattr(self.data_loader, attr)
        ]
        if missing_attributes:
            raise AttributeError(
                f"Data loader is missing required attributes: {', '.join(missing_attributes)}"
            )
        _logger.debug("Data loader validation done.")

    def build_model(self) -> None:
        _logger.info("Model building ...")
        self.model = Sequential(
            [
                Conv2D(
                    64,
                    (3, 3),
                    activation="relu",
                    input_shape=(*self.data_loader.img_size, 3),
                    kernel_initializer="he_normal",
                ),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal"),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(256, activation="relu", kernel_initializer="he_normal"),
                Dropout(0.5),
                Dense(128, activation="relu", kernel_initializer="he_normal"),
                Dense(self.data_loader.num_classes, activation="softmax"),
            ]
        )
        _logger.info("Model built successfully.")

    def compile_model(self) -> None:
        if self.model is None:
            raise ValueError("Model must be built before it can be compiled.")
        if not self.model.compiled:
            _logger.info("Compiling model ...")
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            _logger.info("Model compiled successfully.")
        else:
            _logger.warning("Model has already been compiled.")

    def train_model(self) -> None:
        if self.model is None:
            raise ValueError("Model must be built and compiled before training.")

        _logger.info("Starting model training with augmentations and callbacks ...")
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model_checkpoint = ModelCheckpoint(
            "best_model.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
        self.training_results = self.model.fit(
            self.data_loader.train_data,
            validation_data=self.data_loader.val_data,
            epochs=self.epochs,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
        )

        _logger.info("Model training complete.")

    def evaluate(self) -> float:
        _, val_accuracy = self.model.evaluate(self.data_loader.val_data)
        _logger.info(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        return val_accuracy

    def save_model(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        _logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        if path.is_file() and path.suffix == ".keras":
            self.model = tf.keras.models.load_model(path)
            _logger.info(f"Model loaded from {path}")
        else:
            _logger.error(
                f"Invalid file path or the file is not a valid .keras file: {path}"
            )
            raise OSError(
                f"File at {path} is not a valid .keras file or does not exist."
            )
