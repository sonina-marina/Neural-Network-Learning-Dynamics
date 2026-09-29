import tensorflow as tf
from tensorflow import keras


def create_linear_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])

    return model


def create_nonlinear_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=[1]),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    return model


def create_surface_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=[2]),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    return model
