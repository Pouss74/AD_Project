import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.dates import date2num

def generate_asset_price_graph(asset_name, start_date, end_date):
    # Load the data
    data = pd.read_csv('DataCapstone.csv', delimiter=';', decimal=',')

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', dayfirst=True)  # Ensure the date format is correct

    # Filter data by date
    mask = (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))
    filtered_data = data.loc[mask]

    # Check if the asset exists in the columns
    if asset_name not in filtered_data.columns:
        raise ValueError(f"The asset {asset_name} does not exist in the data.")

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_data['Date'], filtered_data[asset_name], label=asset_name)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Price of {asset_name} from {start_date} to {end_date}')
    plt.legend()
    plt.grid(True)

    # Limit the number of ticks on the y-axis
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit to 10 ticks

    # Adjust layout to ensure labels are not cut off
    plt.tight_layout()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf

def generate_rescaled_plot(start_date, end_date):
    # Load the data specifying the column separator
    data = pd.read_csv('DataCapstone.csv', delimiter=';', decimal='.')

    # Strip any extra spaces from the column names
    data.columns = data.columns.str.strip()

    # Convert the data in the columns to float after replacing commas with periods and removing spaces
    data['S&P 500 PRICE IN USD'] = data['S&P 500 PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)
    data['GOLD PRICE IN USD'] = data['GOLD PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)
    data['BITCOIN PRICE IN USD'] = data['BITCOIN PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)
    data['ETHEREUM PRICE IN USD'] = data['ETHEREUM PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)

    # Convert the 'Date' column to datetime type
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

    # Filter data between start_date and end_date
    mask = (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))
    filtered_data = data.loc[mask]

    # Create a single plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Ethereum, S&P 500, and Gold on the primary y-axis
    ax1.plot(filtered_data['Date'], filtered_data['S&P 500 PRICE IN USD'], color='black', label='S&P 500')
    ax1.plot(filtered_data['Date'], filtered_data['GOLD PRICE IN USD'], color='gold', label='Gold')
    ax1.plot(filtered_data['Date'], filtered_data['ETHEREUM PRICE IN USD'], color='lightgreen', label='Ethereum')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price in USD')
    ax1.tick_params(axis='y')

    # Create a secondary y-axis for Bitcoin
    ax2 = ax1.twinx()
    ax2.plot(filtered_data['Date'], filtered_data['BITCOIN PRICE IN USD'], color='green', label='Bitcoin (secondary axis)')
    ax2.set_ylabel('Bitcoin Price in USD')
    ax2.tick_params(axis='y')

    # Add a legend with all labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Title and grid
    plt.title('Price Evolution of S&P 500, Gold, Ethereum, and Bitcoin')
    plt.grid(True)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return buf

def generate_correlation_matrix():
    # Load the data
    data = pd.read_csv('DataCapstone.csv', delimiter=';', decimal=',')

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', dayfirst=True)  # Ensure the date format is correct

    # Calculate the correlation matrix excluding the Date column
    correlation_matrix = data.drop(columns=['Date']).corr()

    # Plot the correlation matrix with color coding
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix with Color Coding')

    # Adjust layout to ensure labels are not cut off
    plt.tight_layout()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf

def load_and_prepare_data():
    # Load the data specifying the column separator
    data = pd.read_csv('DataCapstone.csv', delimiter=';', decimal='.')

    # Strip any extra spaces from the column names
    data.columns = data.columns.str.strip()

    # Convert the data in the columns to float after replacing commas with periods and removing spaces
    data['S&P 500 PRICE IN USD'] = data['S&P 500 PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)
    data['GOLD PRICE IN USD'] = data['GOLD PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)
    data['BITCOIN PRICE IN USD'] = data['BITCOIN PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)
    data['ETHEREUM PRICE IN USD'] = data['ETHEREUM PRICE IN USD'].str.replace(' ', '').str.replace(',', '.').astype(float)

    # Convert the 'Date' column to datetime type
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

    # Calculate simple returns as the percentage change between consecutive prices
    data['S&P 500 RETURN'] = (data['S&P 500 PRICE IN USD'] / data['S&P 500 PRICE IN USD'].shift(1) - 1) * 100
    data['GOLD RETURN'] = (data['GOLD PRICE IN USD'] / data['GOLD PRICE IN USD'].shift(1) - 1) * 100
    data['BITCOIN RETURN'] = (data['BITCOIN PRICE IN USD'] / data['BITCOIN PRICE IN USD'].shift(1) - 1) * 100
    data['ETHEREUM RETURN'] = (data['ETHEREUM PRICE IN USD'] / data['ETHEREUM PRICE IN USD'].shift(1) - 1) * 100

    return data

def generate_plot(data, asset, start_date, end_date):
    # Filter data between start_date and end_date
    mask = (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))
    filtered_data = data.loc[mask]

    # Map asset names to column names
    asset_column = {
        'S&P 500': 'S&P 500 RETURN',
        'GOLD': 'GOLD RETURN',
        'BITCOIN': 'BITCOIN RETURN',
        'ETHEREUM': 'ETHEREUM RETURN'
    }

    if asset not in asset_column:
        raise ValueError(f"The asset {asset} is not valid. Please choose from 'S&P 500', 'GOLD', 'BITCOIN', or 'ETHEREUM'.")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the selected asset's simple returns
    ax.plot(filtered_data['Date'], filtered_data[asset_column[asset]], label=f'{asset} Returns')
    ax.set_title(f'{asset} Returns')
    ax.set_ylabel('Simple Return (%)')
    ax.set_xlabel('Date')
    ax.grid(True)
    ax.legend()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return buf

def plot_regression(data, asset, start_date, end_date, log=False):
    # Map asset names to column names
    asset_column = {
        'S&P 500': 'S&P 500 PRICE IN USD',
        'GOLD': 'GOLD PRICE IN USD',
        'BITCOIN': 'BITCOIN PRICE IN USD',
        'ETHEREUM': 'ETHEREUM PRICE IN USD'
    }
    
    # Filter data between start_date and end_date
    mask = (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))
    filtered_data = data.loc[mask]

    # Convert date to ordinal numbers for regression analysis
    dates = date2num(filtered_data['Date'])
    prices = filtered_data[asset_column[asset]]

    if log:
        prices = np.log(prices)
        y_label = 'Log Price in USD'
    else:
        y_label = 'Price in USD'

    # Reshape data for scikit-learn
    x = dates.reshape(-1, 1)
    y = prices.values.reshape(-1, 1)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(x, y)
    
    # Predict values
    y_pred = model.predict(x)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the actual data
    ax.plot(filtered_data['Date'], y, color='black', label=f'{asset} Price')
    
    # Overlay the regression line
    ax.plot(filtered_data['Date'], y_pred, color='red', linewidth=2, linestyle='--', label=f'{asset} Regression Line')
    ax.set_title(f'{asset} {"Log " if log else ""}Price Evolution with Regression Line')
    ax.set_xlabel('Date')
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True)

    # Adjust the x-axis to start from the specific date
    ax.set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return buf
