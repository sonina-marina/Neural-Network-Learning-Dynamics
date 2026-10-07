import numpy as np

from src.functions import sphere_function
from src.models import create_surface_model
from src.training import train_model
from src.visualisation import plot_surface_training

x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)

X, Y = np.meshgrid(x, y)
Z = sphere_function(X, Y)

inputs = np.column_stack((X.ravel(), Y.ravel()))
targets = Z.ravel()


model = create_surface_model()

# Initial prediction
Z_initial = model.predict(inputs, verbose=0).reshape(X.shape)

history = train_model(
    model,
    inputs,
    targets,
    epochs=1000,
    learning_rate=0.005
)

# Final prediction
Z_final = model.predict(inputs, verbose=0).reshape(X.shape)


plot_surface_training(
    X=X,
    Y=Y,
    Z_target=Z,
    Z_initial=Z_initial,
    Z_final=Z_final,
    loss_history=history.losses
)
