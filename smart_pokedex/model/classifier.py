import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

from smart_pokedex.model.data_loader import PokemonImagesDataLoader

_logger = logging.getLogger(__name__)


class PokemonClassifier:

    def __init__(
        self,
        data_loader: PokemonImagesDataLoader | None = None,
        epochs: int = 20,
        img_size: tuple[int, int] = (128, 128),
        batch_size: int = 32,
    ) -> None:
        self.epochs = epochs
        self.data_loader = data_loader
        self.img_size = img_size
        self.batch_size = batch_size
        self.training_results = None
        self.model = None

    def validate_data_loader(self) -> None:
        """
        Validate that the data loader has the necessary data attributes.
        """
        _logger.debug("Validating data loader attributes...")
        required_attributes = ["train_data", "test_data", "pokemon_class_indices"]
        missing_attributes = [
            attr for attr in required_attributes if not hasattr(self.data_loader, attr)
        ]
        if missing_attributes:
            raise AttributeError(
                f"Data loader is missing required attributes: {', '.join(missing_attributes)}"
            )
        _logger.debug("Data loader validated successfully.")

    def build_model(self) -> None:
        """
        Build an improved CNN model for classification.
        """
        _logger.info("Building the model...")
        self.validate_data_loader()

        self.model = Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=(*self.img_size, 3)),
                BatchNormalization(),
                MaxPooling2D(),
                Conv2D(64, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D(),
                Conv2D(128, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D(),
                Conv2D(256, (3, 3), activation="relu"),
                BatchNormalization(),
                MaxPooling2D(),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(
                    len(self.data_loader.pokemon_class_indices), activation="softmax"
                ),
            ]
        )

        _logger.info("Model built successfully.")

    def compile_model(self) -> None:
        """
        Compile the model with the Adam optimizer and categorical crossentropy loss.
        """
        if self.model is None:
            raise ValueError("Model must be built before it can be compiled.")

        _logger.info("Compiling the model...")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        _logger.info("Model compiled successfully.")

    def train_model(self) -> None:
        """
        Train the model with the specified callbacks.
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training.")

        _logger.info("Starting model training with augmentations and callbacks...")

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
            validation_data=self.data_loader.test_data,
            epochs=self.epochs,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
        )

        _logger.info("Model training complete.")

    def evaluate(self) -> float:
        """
        Evaluate the model on the test data.
        """
        if self.model is None:
            raise ValueError("Model must be built before evaluation.")

        _logger.info("Evaluating model...")
        _, test_accuracy = self.model.evaluate(self.data_loader.test_data)
        _logger.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        return test_accuracy

    def save_model(self, path: Path) -> None:
        """
        Save the trained model to a specified path.
        """
        if self.model is None:
            raise ValueError("Model must be built before saving.")

        self.model.save(path)
        _logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """
        Load a trained model from a specified path.
        """
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
