from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


class Predictor(ABC):
    @abstractmethod
    def predict(self, *args, **kwargs) -> None:
        pass


class PokemonPredictor(Predictor):
    def __init__(
        self, model: tf.keras.Model, pokemon_class_indices: dict[str, int]
    ) -> None:
        self.model = model
        self.pokemon_class_indices = pokemon_class_indices

    def predict(self, img_path: str) -> str:
        img = image.load_img(img_path, target_size=self.model.input_shape[1:3])
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        class_idx = np.argmax(prediction, axis=-1)[0]
        pokemon_name = self.pokemon_class_indices.get(class_idx)
        return pokemon_name
