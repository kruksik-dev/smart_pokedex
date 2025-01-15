import tensorflow as tf

from smart_pokedex.model.image_objects import ImageData, PokemonImagesData


def test_image_data_initialization() -> None:
    train_data = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    test_data = tf.data.Dataset.from_tensor_slices([4, 5, 6])

    image_data = ImageData(train_data=train_data, test_data=test_data)

    assert image_data.train_data == train_data
    assert image_data.test_data == test_data


def test_pokemon_images_data_initialization() -> None:
    train_data = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    test_data = tf.data.Dataset.from_tensor_slices([4, 5, 6])
    pokemon_class_indices = {0: "Pikachu", 1: "Bulbasaur"}

    pokemon_images_data = PokemonImagesData(
        train_data=train_data,
        test_data=test_data,
        pokemon_class_indices=pokemon_class_indices,
    )

    assert pokemon_images_data.train_data == train_data
    assert pokemon_images_data.test_data == test_data
    assert pokemon_images_data.pokemon_class_indices == pokemon_class_indices
