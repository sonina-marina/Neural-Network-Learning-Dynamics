import matplotlib.pyplot as plt
import numpy as np


def plot_linear_training(
    x,
    target_function,
    initial_weight,
    initial_bias,
    final_weight,
    final_bias,
    loss_history
):
    y_target = target_function(x)

    y_initial = initial_weight * x + initial_bias
    y_final = final_weight * x + final_bias

    epochs = np.arange(1, len(loss_history) + 1)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)

    plt.title("Function approximation")

    plt.plot(x, y_target, label="Target function")
    plt.plot(x, y_initial, label="Initial model")
    plt.plot(x, y_final, label="Trained model")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.title("Loss during training")
    plt.plot(epochs, loss_history)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()

    plt.tight_layout()
    plt.show()
