import streamlit as st
import pandas as pd
from backend import generate_asset_price_graph, generate_normalized_graph, generate_correlation_matrix

# Apply custom CSS for a black background with Bitcoin, Ethereum, and blockchain graphics
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000; /* Black background color */
        color: #ffffff; /* White text color for contrast */
        background-image: url('https://example.com/bitcoin_graphic.png'), url('https://example.com/ethereum_graphic.png'), url('https://example.com/blockchain_graphic.png');
        background-position: left top, right top, center bottom; /* Position the graphics */
        background-repeat: no-repeat, no-repeat, no-repeat; /* Do not repeat the graphics */
        background-size: 100px, 100px, 100px; /* Size of the graphics */
    }
    .css-18e3th9 {
        background-color: #000000; /* Background color for main container */
    }
    .css-1d391kg { 
        background-color: #1a1a1a; /* Slightly lighter black for the sidebar */
    }
    .css-1d391kg header, .css-1d391kg footer {
        background-color: #000000; /* Match header and footer background to app */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to get forecast image path based on selection
def get_forecast_image_path(asset, model):
    # Define the mapping of asset and model to image file paths
    forecast_images = {
        "Bitcoin": {
            "ARIMA": "forecast/Bitcoin Forecast ARIMA Model.png",
            "LSTM": "forecast/Bitcoin Forecast LSTM Model.png",
            "LSTM with correlation": "forecast/Bitcoin Forecast LSTM with Correlation.png"
        },
        "Ethereum": {
            "ARIMA": "forecast/Ethereum Forecast ARIMA Model.png",
            "LSTM": "forecast/Ethereum Forecast LSTM Model.png",
            "LSTM with correlation": "forecast/Ethereum Forecast LSTM with Correlation.png"
        }
    }
    return forecast_images[asset][model]


# App title
st.title("Future is Yours")

# Create tabs
tab1, tab2 = st.tabs(["Historical Data", "Forecasts"])

# Historical Data tab
with tab1:
    st.header("Historical Data")

    # Input for asset price graph
    asset_name = st.selectbox("Select an asset", ["S&P 500 PRICE IN USD", "GOLD PRICE IN USD", "BITCOIN PRICE IN USD",
                                                  "ETHEREUM PRICE IN USD"])
    start_date = st.date_input("Start date", value=pd.to_datetime("2019-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2019-12-31"))

    if st.button("Show Asset Price"):
        buf = generate_asset_price_graph(asset_name, start_date, end_date)
        st.image(buf, use_column_width=True)

    # Button to show normalized graph
    if st.button("Show Normalized Prices"):
        buf = generate_normalized_graph(start_date, end_date)
        st.image(buf, use_column_width=True)

    # Button to show correlation matrix
    if st.button("Show Correlation Matrix"):
        buf = generate_correlation_matrix()
        st.image(buf, use_column_width=True)

# Forecasts tab
with tab2:
    st.header("Forecasts")

    # Dropdowns to select asset and model
    forecast_asset = st.selectbox("Select an asset for forecast", ["Bitcoin", "Ethereum"])
    forecast_model = st.selectbox("Select a model", ["ARIMA", "LSTM", "LSTM with correlation"])

    # Button to display forecast
    if st.button("Show Forecast"):
        image_path = get_forecast_image_path(forecast_asset, forecast_model)
        st.image(image_path, caption=f"{forecast_asset} forecast using {forecast_model}", use_column_width=True)
