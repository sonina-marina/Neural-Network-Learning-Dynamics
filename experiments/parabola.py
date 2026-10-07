import numpy as np

from src.functions import quadratic_function
from src.models import create_nonlinear_model
from src.training import train_model
from src.visualisation import plot_nonlinear_training

xs = np.array([-2, -1, 0, 1, 2, 4], dtype=float)
ys = quadratic_function(xs)

model = create_nonlinear_model()

y_initial = model.predict(xs, verbose=0).flatten()

history = train_model(
    model,
    xs,
    ys,
    epochs=100,
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
