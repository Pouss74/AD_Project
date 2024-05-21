import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
from matplotlib.ticker import MaxNLocator


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
    data = pd.read_csv('DataCapstone.csv', delimiter=';', decimal=' ')

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