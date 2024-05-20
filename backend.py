import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from matplotlib.dates import date2num
from sklearn.linear_model import LinearRegression
import io
from matplotlib.ticker import MaxNLocator

# Function to load and prepare data
def load_and_prepare_data(file_path='DataCapstone.csv'):
    data = pd.read_csv(file_path, delimiter=';', decimal=',')
    data.columns = data.columns.str.strip()
    data['S&P 500 PRICE IN USD'] = data['S&P 500 PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)
    data['GOLD PRICE IN USD'] = data['GOLD PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)
    data['BITCOIN PRICE IN USD'] = data['BITCOIN PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)
    data['ETHEREUM PRICE IN USD'] = data['ETHEREUM PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    return data

# Function to calculate simple returns and generate a graph
def generate_simple_returns_graph():
    data = load_and_prepare_data()
    data['S&P 500 RETURN'] = (data['S&P 500 PRICE IN USD'] / data['S&P 500 PRICE IN USD'].shift(1) - 1) * 100
    data['GOLD RETURN'] = (data['GOLD PRICE IN USD'] / data['GOLD PRICE IN USD'].shift(1) - 1) * 100
    data['BITCOIN RETURN'] = (data['BITCOIN PRICE IN USD'] / data['BITCOIN PRICE IN USD'].shift(1) - 1) * 100
    data['ETHEREUM RETURN'] = (data['ETHEREUM PRICE IN USD'] / data['ETHEREUM PRICE IN USD'].shift(1) - 1) * 100
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    axes[0, 0].plot(data['Date'], data['S&P 500 RETURN'], color='black')
    axes[0, 0].set_title('S&P 500 Returns')
    axes[0, 0].set_ylabel('Simple Return (%)')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(data['Date'], data['GOLD RETURN'], color='gold')
    axes[0, 1].set_title('Gold Returns')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(data['Date'], data['BITCOIN RETURN'], color='green')
    axes[1, 0].set_title('Bitcoin Returns')
    axes[1, 0].set_ylabel('Simple Return (%)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(data['Date'], data['ETHEREUM RETURN'], color='lightgreen')
    axes[1, 1].set_title('Ethereum Returns')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Function to generate a linear regression graph
def generate_linear_regression_graph():
    data = load_and_prepare_data()
    start_date = pd.Timestamp('2019-01-01')
    data = data[data['Date'] >= start_date]
    dates = date2num(data['Date'])
    
    def plot_regression(x, y, ax, label, color):
        x = x.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        ax.plot(data['Date'], y, color=color, label=f'{label} Price')
        ax.plot(data['Date'], y_pred, color='red', linewidth=2, linestyle='--', label=f'{label} Regression Line')
        ax.set_title(f'{label} Price Evolution with Regression Line')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price in USD')
        ax.legend()
        ax.set_xlim([start_date, data['Date'].max()])

    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    plot_regression(dates, data['S&P 500 PRICE IN USD'], axs[0, 0], 'S&P 500', 'black')
    plot_regression(dates, data['GOLD PRICE IN USD'], axs[0, 1], 'Gold', 'gold')
    plot_regression(dates, data['BITCOIN PRICE IN USD'], axs[1, 0], 'Bitcoin', 'green')
    plot_regression(dates, data['ETHEREUM PRICE IN USD'], axs[1, 1], 'Ethereum', 'lightgreen')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Function to generate a log-linear regression graph
def generate_log_linear_regression_graph():
    data = load_and_prepare_data()
    start_date = pd.Timestamp('2019-01-01')
    data = data[data['Date'] >= start_date]
    data['Log S&P 500'] = np.log(data['S&P 500 PRICE IN USD'])
    data['Log GOLD'] = np.log(data['GOLD PRICE IN USD'])
    data['Log BITCOIN'] = np.log(data['BITCOIN PRICE IN USD'])
    data['Log ETHEREUM'] = np.log(data['ETHEREUM PRICE IN USD'])
    dates = date2num(data['Date'])

    def plot_regression(x, y, ax, label, color):
        x = x.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        ax.plot(data['Date'], y, color=color, label=f'{label} Price')
        ax.plot(data['Date'], y_pred, color='red', linewidth=2, linestyle='--', label=f'{label} Regression Line')
        ax.set_title(f'{label} Log Price Evolution with Regression Line')
        ax.set_xlabel('Date')
        ax.set_ylabel('Log Price in USD')
        ax.legend()
        ax.set_xlim([start_date, data['Date'].max()])

    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    plot_regression(dates, data['Log S&P 500'], axs[0, 0], 'S&P 500', 'black')
    plot_regression(dates, data['Log GOLD'], axs[0, 1], 'Gold', 'gold')
    plot_regression(dates, data['Log BITCOIN'], axs[1, 0], 'Bitcoin', 'green')
    plot_regression(dates, data['Log ETHEREUM'], axs[1, 1], 'Ethereum', 'lightgreen')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
