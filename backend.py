import pandas aimport matplotlib.pyplot as plt
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


def generate_normalized_graph(start_date, end_date):
    # Load the data
    data = pd.read_csv('DataCapstone.csv', delimiter=';', decimal=',')

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', dayfirst=True)  # Ensure the date format is correct

    # Filter data by date
    mask = (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))
    filtered_data = data.loc[mask]

    # List of assets to plot
    assets = ['S&P 500 PRICE IN USD', 'GOLD PRICE IN USD', 'BITCOIN PRICE IN USD', 'ETHEREUM PRICE IN USD']

    # Normalize the values relative to their values at the start date
    for asset in assets:
        filtered_data[asset] = filtered_data[asset] / filtered_data[asset].iloc[0]

    # Create the plot
    plt.figure(figsize=(10, 5))
    for asset in assets:
        plt.plot(filtered_data['Date'], filtered_data[asset], label=asset)

    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.title(f'Normalized Prices of Assets from {start_date} to {end_date}')
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