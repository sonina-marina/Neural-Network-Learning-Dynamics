import numpy as np

from src.functions import linear_function
from src.models import create_linear_model
from src.training import train_model
from src.visualisation import plot_linear_training


xs = np.array([-1, 0, 1, 2, 4], dtype=float)
ys = linear_function(xs)

model = create_linear_model()

history = train_model(
    model,
    xs,
    ys,
    epochs=100,
    learning_rate=0.01
)

#print(history.losses)
#print(history.weights)
#print(history.biases)

x_test = np.array([0.5])
prediction = model.predict(x_test, verbose=0)[0][0]
print(f"x = {x_test[0]}")
print(f"Prediction = {prediction}")
print(f"Target = {linear_function(x_test)[0]}")

initial_weight = history.weights[0][0][0][0]
initial_bias = history.biases[0][0][0]

final_weight = history.weights[-1][0][0][0]
final_bias = history.biases[-1][0][0]

plot_linear_training(
    xs, 
    linear_function,
    initial_weight,
    initial_bias,
    final_weight,
    final_bias,
    history.losses
)
