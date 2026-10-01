import numpy as np

from src.functions import sin_function
from src.models import create_nonlinear_model
from src.training import train_model
from src.visualisation import plot_nonlinear_training

xs = np.linspace(-2 * np.pi, 2 * np.pi, 100)
ys = sin_function(xs)

model = create_nonlinear_model()

y_initial = model.predict(xs, verbose=0).flatten()

history = train_model(
    model,
    xs,
    ys,
    epochs=1000,
    learning_rate=0.005
)

#print(history.losses)

y_final = model.predict(xs, verbose=0).flatten()

plot_nonlinear_training(
    x=xs,
    y_target=ys,
    y_initial=y_initial,
    y_final=y_final,
    loss_history=history.losses
)
