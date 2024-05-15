import pandas as pd
import matplotlib.pyplot as plt
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

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf
