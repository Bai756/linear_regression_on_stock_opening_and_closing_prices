import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter.simpledialog import askstring
from tkinter import messagebox

def compute_error(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points.iloc[i]['date']
        y = points.iloc[i]['open']
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
        x = points.iloc[i]['date']
        y = points.iloc[i]['open']
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def plot_points(points, b, m, symbol):
    x = points['date']
    y = points['open']
    plt.scatter(x, y)
    plt.plot(x, m * x + b, color='red')
    plt.xlabel('Days since 2010-01-01')
    plt.ylabel('Open Price')

    plt.title('Linear Regression of' + ' ' + symbol)
    plt.show()

def convert_dates_to_days(dates):
    start_date = pd.Timestamp('2010-01-01')
    return (pd.to_datetime(dates) - start_date).dt.days

def run(company_symbol):
    start_time = time.time()

    points = pd.read_csv("prices-split-adjusted.csv")
    symbol = company_symbol
    company_data = points[points['symbol'] == symbol]

    company_data = company_data[['symbol', 'open', 'date']]

    company_data['date'] = convert_dates_to_days(company_data['date'])

    # Defining the hyperparameters
    learning_rate = 0.0000001
    initial_b = 0
    initial_m = 0
    num_iterations = 100
    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error(initial_b, initial_m, company_data)}")

    [b, m] = gradient_descent_runner(company_data, initial_b, initial_m, learning_rate, num_iterations)
    print(f"After {num_iterations} iterations b = {b}, m = {m}, error = {compute_error(b, m, company_data)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")

    plot_points(company_data, b, m, symbol)

def ask_company_symbol():
    root = tk.Tk()
    root.withdraw()
    
    points = pd.read_csv("prices-split-adjusted.csv")
    valid_symbols = points['symbol'].unique()

    company_symbol = tk.simpledialog.askstring("Input", "Enter the company symbol (e.g., AMZN):")
    if company_symbol is not None:
        company_symbol = company_symbol.upper()
    if company_symbol is None:
        tk.messagebox.showerror("Error", "No company symbol entered. Please try again.")
    elif company_symbol in valid_symbols:
        root.destroy()
        run(company_symbol)
    else:
        tk.messagebox.showerror("Error", "Invalid company symbol. Please try again.")
    

if __name__ == '__main__':
    ask_company_symbol()