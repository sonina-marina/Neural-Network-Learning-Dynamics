# Neural Network Learning Dynamics

This project explores how neural networks learn to approximate mathematical functions and how different training parameters affect the learning process.

The main idea is to provide a set of points generated from a known mathematical function and train a neural network to approximate the relationship between the input and output values. The trained model is then compared with the original function to observe how well the network has learned it.

The project includes experiments with four types of functions:

* Linear function: $y = 3x + 1$
* Quadratic function: $y = x^2$
* Sine function: $y = \sin(x)$
* Paraboloid: $z = x^2 + y^2$

The project focuses not only on the final approximation, but also on the dynamics of the training process. The experiments make it possible to observe how the model changes during training, how the loss evolves over epochs, and how the choice of learning rate affects convergence.

## Project Structure

The project is divided into several modules:
* functions.py contains the mathematical functions used as target functions.
* models.py contains the neural network architectures for different types of functions.
* training.py is responsible for model training and collecting information about the training process, including loss, weights, and biases.
* visualisation.py contains functions for visualizing the target functions, model predictions, and training dynamics.
* experiments/ contains separate experiments for the linear function, nonlinear functions, and the 3D surface.

This separation allows the same training procedure to be reused with different models and functions.

## Function Approximation

The training data consists of points generated from a known mathematical function. For example, for the linear function $y = 3x + 1$, a set of x-values is generated and the corresponding y-values are calculated.

The neural network receives these points as training data and learns an approximation of the underlying relationship: $x \rightarrow \hat{y}$.
The predicted values $\hat{y}$ are then compared with the original values y using the mean squared error (MSE) loss function.

For the paraboloid, the model receives two input variables: (x,y) \rightarrow \hat{z}, and learns to approximate $z=x^2+y^2$.

This experiment extends the same idea from a one-dimensional function to a two-dimensional surface.

## Visualisation

For the linear function, three aspects of the training process are visualized:
1. Function approximation - the target function, the initial model prediction, and the final trained model prediction are shown on the same graph.
2. Loss during training - the MSE loss is plotted against the number of epochs.
3. Weight changes - the value of the model’s weight is plotted against the number of epochs and compared with the target value w=3.

The weight visualization is particularly useful for the linear model because the network has only two parameters: $\hat{y}=wx+b$.
Therefore, the weight directly determines the slope of the learned line.

Nonlinear functions
For the quadratic and sine functions, the visualisation focuses on:
* the target function and model approximation;
* the loss during training.

The internal weights of these networks are not visualized individually because the nonlinear models contain multiple layers and many parameters, making their individual values much less directly interpretable.

Paraboloid

For the two-dimensional function, the target and predicted functions are visualized as 3D surfaces. This makes it possible to compare the original surface with the one learned by the neural network.

## Training Dynamics

One of the interesting observations in the experiments is that the loss does not always decrease smoothly.
For some combinations of model architecture and learning rate, the loss can decrease rapidly at first and then begin to oscillate.

This behavior can occur when the learning rate is too large. The learning rate controls the size of the parameter update during gradient descent. A large learning rate can cause the model to make large steps and repeatedly move past the region of the minimum loss instead of approaching it smoothly.

The learning rate also affects the trajectory of individual parameters. In the linear experiment, the learned weight does not necessarily approach the target value w=3 monotonically. It can cross the optimal value and then approach it from the other side.

For example:
$2.5 \rightarrow 2.8 \rightarrow 3.1 \rightarrow 3.05 \rightarrow 3.01 \rightarrow 3.00$

Thus, the weight can overshoot the optimal value before converging toward it.

This demonstrates why observing only the final result is not enough to fully understand the training process. Tracking the loss and parameter values over time provides additional insight into how the neural network actually learns.

## Results and Observations

1. Linear Function

Figure 1. Linear function approximation after 100 epochs with a learning rate of 0.01. The model has started approaching the target function, while the learned weight is still noticeably different from the optimal value w=3.
<img width="993" height="708" alt="image" src="https://github.com/user-attachments/assets/3e702a85-0ea3-4444-a9df-a7147e6a33c1" />

Figure 2. Linear function approximation after 500 epochs with a learning rate of 0.01. The learned line is closer to the target function, and the weight gradually approaches the optimal value w=3.
<img width="995" height="709" alt="image" src="https://github.com/user-attachments/assets/99067d50-308f-4f9b-a8c3-ce76e9d3d70b" />

Figure 3. Linear function approximation after 1000 epochs with a learning rate of 0.01. The trained model closely approximates the target function, while the loss approaches a low value and the weight converges toward w=3.
<img width="986" height="705" alt="image" src="https://github.com/user-attachments/assets/9e3032db-2c59-4c53-b0c1-29e4d0113536" />

Figure 4. Effect of the learning rate on linear model training. With a higher learning rate of 0.1, the model makes larger parameter updates, which can cause stronger oscillations around the minimum loss. With a learning rate of 0.01, the convergence is smoother.

Figure 5. Quadratic function approximation after 100 epochs. The neural network has started learning the nonlinear relationship, but the approximation is still relatively rough.

