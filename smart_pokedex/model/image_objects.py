from dataclasses import dataclass

import tensorflow as tf


@dataclass
class ImageData:
    train_data: tf.data.Dataset
    test_data: tf.data.Dataset


@dataclass
class PokemonImagesData(ImageData):
    pokemon_class_indices: dict[int, str]
