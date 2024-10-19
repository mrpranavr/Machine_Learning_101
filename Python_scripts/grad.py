# Gradient descent for linear regression
# y = wx + b
# loss = (y - wx - c)^2 / N , ie., (y - yhat)^2 / N (Mean Sqaured Error)

import numpy as np

# Initialise some parameters
x = np.random.randn(10, 1)
y = (3 * x) + np.random.rand()

# Parameters
w = 0.0 # Ideally after GD, value should be 2
b = 0.0 # Ideally after GD, value should be np.random.rand() part of the above y equation

# Hyperparamters
learning_rate = 0.01
num_iterations = 1000

# Create gradient descent function
def gradient_descent(x, y, w, b, learning_rate, num_iterations):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]
    # loss = (y - (wx + b)) ^ 2 
    for xi, yi in zip(x, y):
        dldw += -2 * xi * (yi - (w * xi + b)) 
        dldb += -2 * (yi - (w * xi + b))

    w = w - (learning_rate / N) * dldw
    b = b - (learning_rate / N) * dldb

    return w, b

# Run gradient descent
for i in range(num_iterations):
    w, b = gradient_descent(x, y, w, b, learning_rate, num_iterations)
    yhat = w * x + b
    loss = np.mean((y - yhat) ** 2)
    # print(f"Iteration {i}: w = {w}, b = {b}, loss = {loss}")
    if i % 10 == 0:
        print(f"Iteration {i}: w = {w}, b = {b}, loss = {loss}")