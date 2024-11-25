import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
import time

def compute_error(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points.iloc[i, 1]
        y = points.iloc[i, 2]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
    return [b, m]

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points.iloc[i, 1]
        y = points.iloc[i, 2]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def plot_points(points, b, m):
    x = points['open']
    y = points['close']
    plt.scatter(x, y)
    plt.plot(x, m * x + b, color='red')
    plt.xlabel('Open Price')
    plt.ylabel('Close Price')
    plt.show()

def run():
    start_time = time.time()

    points = pd.read_csv("/Users/qiaoe27/programs/linear regression/prices-split-adjusted.csv")
    points = points[['symbol', 'open', 'close']]

    # Filter the DataFrame for a specific company
    company_name = 'AMZN'
    company_data = points[points['symbol'] == company_name]

    # defining the hyperparameters
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1
    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error(initial_b, initial_m, points)}")

    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(f"After {num_iterations} iterations b = {b}, m = {m}, error = {compute_error(b, m, points)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")

    plot_points(points, b, m)

if __name__ == '__main__':
    run()