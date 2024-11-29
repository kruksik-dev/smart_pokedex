import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import tensorflow as tf

from smart_pokedex.model.data_loader import PokemonImagesData

_logger = logging.getLogger(__name__)


class Classifier(ABC):
    """
    Abstract base class for classifiers.

    This class defines the common interface for building, training, evaluating,
    saving, and loading machine learning models.
    """

    @abstractmethod
    def build_model(self) -> None:
        """
        Abstract method to build a machine learning model.
        """
        pass

    @abstractmethod
    def compile_model(self) -> None:
        """
        Abstract method to compile the model with the appropriate optimizer and loss function.
        """
        pass

    @abstractmethod
    def train_model(self) -> None:
        """
        Abstract method to train the model.
        """
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """
        Abstract method to evaluate the model on test data.

        Returns:
            float: The test accuracy.
        """
        pass

    @abstractmethod
    def save_model(self, path: Path) -> None:
        """
        Abstract method to save the trained model to disk.

        Args:
            path (Path): The location to save the model.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """
        Abstract method to load a trained model from disk.

        Args:
            model_path (Path): The location of the saved model.
        """
        pass


class PokemonClassifier(Classifier):
    """
    A classifier for Pokémon image datasets.

    This class provides functionality for building, compiling, training,
    evaluating, and saving/loading models specifically for classifying Pokémon images.

    Attributes:
        data (PokemonImagesData | None): The dataset containing the training and test data.
        epochs (int): The number of epochs for training.
        img_size (tuple[int, int]): The size to which input images are resized.
        batch_size (int): The batch size for training and evaluation.
        model (tf.keras.Model | None): The TensorFlow model.
        pokemon_class_indices (Dict[int, str]): Mapping of class indices to Pokémon species names.
    """

    def __init__(
        self,
        data: PokemonImagesData | None = None,
        epochs: int = 20,
        img_size: tuple[int, int] = (128, 128),
        batch_size: int = 32,
    ) -> None:
        """
        Initializes the Pokémon classifier.

        Args:
            data (PokemonImagesData | None): The dataset containing Pokémon image data. Defaults to None.
            epochs (int): The number of epochs to train the model. Defaults to 20.
            img_size (tuple[int, int]): The target size for resizing images. Defaults to (128, 128).
            batch_size (int): The batch size for training and evaluation. Defaults to 32.
        """
        self.epochs = epochs
        self.data = data
        self.img_size = img_size
        self.batch_size = batch_size
        self.training_results = None
        self.model = None
        self._train_data = getattr(data, "train_data", None)
        self._test_data = getattr(data, "test_data", None)
        self.pokemon_class_indices = getattr(data, "pokemon_class_indices", None)

    def _check_model_initialized(self) -> None:
        """Helper function to check if the model has been initialized."""
        if self.model is None:
            raise ValueError("Model must be built before it can be used.")

    def build_model(self) -> None:
        """
        Build the convolutional neural network (CNN) model for Pokémon classification.

        The model is built using several convolutional layers, batch normalization,
        max pooling, and dense layers.
        """
        _logger.info("Building the model...")
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=(*self.img_size, 3)
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(
                    len(self.pokemon_class_indices), activation="softmax"
                ),
            ]
        )

        _logger.info("Model built successfully.")

    def compile_model(self) -> None:
        """
        Compile the model with the Adam optimizer and categorical crossentropy loss.
        """
        self._check_model_initialized()
        _logger.info("Compiling the model...")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        _logger.info("Model compiled successfully.")

    def train_model(self) -> None:
        """
        Train the model using the training dataset and specified callbacks.
        """
        self._check_model_initialized()
        _logger.info("Starting model training with augmentations and callbacks...")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )

        self.training_results = self.model.fit(
            self._train_data,
            validation_data=self._test_data,
            epochs=self.epochs,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
        )

        _logger.info("Model training complete.")

    def evaluate(self) -> float:
        """
        Evaluate the model on the test dataset.

        Returns:
            float: The test accuracy as a percentage.
        """
        self._check_model_initialized()
        _logger.info("Evaluating model...")
        _, test_accuracy = self.model.evaluate(self._test_data)
        _logger.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        return test_accuracy

    def save_model(self, path: Path) -> None:
        """
        Save the trained model to the specified path.

        Args:
            path (Path): Path where the model will be saved.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._check_model_initialized()
        self.model.save(path)
        _logger.info(f"Model saved to {path}")

    def save_model_with_metadata(self, model_path: Path, metadata_path: Path) -> None:
        """
        Save the trained model and its associated metadata (class indices) to disk.

        Args:
            model_path (Path): Path to save the model file.
            metadata_path (Path): Path to save the metadata (class indices).
        """
        self.save_model(model_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(self.pokemon_class_indices, f, indent=4)
        _logger.info(f"Class indices saved successfully to {metadata_path}")

    def load_model(self, model_path: Path) -> None:
        """
        Load a trained model from the specified path.

        Args:
            model_path (Path): Path to the saved model.
        """
        if not model_path.is_file():
            raise FileNotFoundError(f"Keras model file not found at {model_path=}")
        if model_path.suffix != ".keras":
            raise ValueError(f"Expected .keras file, got {model_path.suffix}")
        self.model = tf.keras.models.load_model(model_path)
        _logger.info("Model loaded successfully from file")

    def load_model_with_metadata(self, model_path: Path, metadata_path: Path) -> None:
        """
        Load both the trained model and its metadata (class indices) from disk.

        Args:
            model_path (Path): Path to the saved model file.
            metadata_path (Path): Path to the metadata (class indices file).
        """
        self.load_model(model_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Json metadata file not found at {metadata_path=}")
        if metadata_path.suffix != ".json":
            raise ValueError(
                f"Expected a json metadata file, got {metadata_path.suffix}"
            )
        try:
            with open(metadata_path, "r") as f:
                class_indices = json.load(f)
            self.pokemon_class_indices = class_indices
            _logger.info(f"Class indices successfully loaded from {metadata_path}")
        except json.JSONDecodeError as e:
            _logger.error(f"Failed to decode JSON from {metadata_path}: {e}")
            raise ValueError(f"Invalid JSON file at {metadata_path}: {e}")

    def _preprocess_image(self, image_path: Path) -> tf.Tensor:
        """
        Preprocess a single image for prediction.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            tf.Tensor: Preprocessed image tensor ready for prediction.
        """
        # Load and decode the image
        image = tf.io.read_file(str(image_path))
        image = tf.image.decode_image(image, channels=3)

        # Resize the image to match the input size of the model
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

        # Add batch dimension for prediction
        return tf.expand_dims(image, axis=0)

    def predict(self, image_path: Path) -> dict:
        """
        Predict the class of a Pokémon image.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            dict: A dictionary containing the predicted class and confidence scores.
        """
        self._check_model_initialized()

        # Preprocess the image
        preprocessed_image = self._preprocess_image(image_path)

        # Perform prediction
        predictions = self.model.predict(preprocessed_image)
        predicted_index = tf.argmax(predictions[0]).numpy()

        # Map index to class name
        if not self.pokemon_class_indices:
            raise ValueError("Class indices are not loaded. Cannot map predictions.")
        predicted_class = self.pokemon_class_indices[str(predicted_index)]

        return {
            "predicted_class": predicted_class,
            # "confidence_scores": confidence_scores.tolist(),
            "image_path": image_path,
        }
