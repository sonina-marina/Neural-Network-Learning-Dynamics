import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class TrainingHistory:
    def __init__(self):
        self.weights = []
        self.biases = []
        self.losses = []


class CustomCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history = TrainingHistory()

    def on_epoch_end(self, epoch, logs=None):
        self.history.losses.append(logs["loss"])

        weights = []
        biases = []

        for layer in self.model.layers:
            layer_weights = layer.get_weights()

            if len(layer_weights) == 2:
                weights.append(layer_weights[0].copy())
                biases.append(layer_weights[1].copy())

        self.history.weights.append(weights)
        self.history.biases.append(biases)


def train_model(model, x, y, epochs, learning_rate=0.01):
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error"
    )

    callback = CustomCallback()

    model.fit(
        x,
        y,
        epochs=epochs,
        callbacks=[callback]
    )

    return callback.history
