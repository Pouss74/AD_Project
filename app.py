import streamlit as st
import pandas as pd
from backend import generate_asset_price_graph, generate_normalized_graph, generate_correlation_matrix

# Apply custom CSS for the desired styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000; /* Black background color */
        color: #ffffff; /* White text color for contrast */
    }
    .css-18e3th9 {
        background-color: #000000; /* Background color for main container */
        color: #ffffff; /* Ensure text is white */
    }
    .css-1d391kg { 
        background-color: #1a1a1a; /* Slightly lighter black for the sidebar */
        color: #ffffff; /* Ensure text is white */
    }
    .css-1d391kg header, .css-1d391kg footer {
        background-color: #000000; /* Match header and footer background to app */
    }
    h1 {
        font-size: 3em; /* Larger font size for the title */
        text-align: center; /* Center align the title */
        font-style: italic; /* Italicize the title */
        color: #ffffff; /* White text for title */
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center; /* Center the tabs */
    }
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important; /* Ensure tab text is white */
    }
    .stButton>button {
        background-color: #ffffff; /* White background for buttons */
        color: #000000; /* Black text for buttons */
        border: none; /* Remove border */
        padding: 10px 20px; /* Add padding */
        margin: 5px; /* Add margin */
        border-radius: 5px; /* Add rounded corners */
        font-size: 16px; /* Increase font size */
        cursor: pointer; /* Pointer cursor on hover */
    }
    .stButton>button:hover {
        background-color: #dddddd; /* Slightly darker on hover */
    }
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .date-input {
        width: 50%;
        margin: 0 auto;
    }
    .css-1q8dd3e, .css-1d391kg, .css-18e3th9, .stSelectbox label, .stDateInput label {
        color: #ffffff !important; /* White text for labels */
    }
    .correlation-text {
        margin-top: 20px;
        line-height: 1.6;
    }
    .regression-text, .css-1q8dd3e, .css-1d391kg, .css-18e3th9, .stSelectbox label, .stRadio label {
        color: #ffffff !important; /* White text for regression tab */
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
st.title("Future is Yours!")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Historical Data", "Re-Scale Graphic", "Returns", "Correlation", "Regression", "ARIMA", "LSTM 1", "LSTM 2"])

# Historical Data tab
with tab1:
    st.header("Historical Data")
    
    # Input for asset price graph
    asset_name = st.selectbox("Select an asset", ["S&P 500 PRICE IN USD", "GOLD PRICE IN USD", "BITCOIN PRICE IN USD",
                                                  "ETHEREUM PRICE IN USD"])
    start_date = st.date_input("Start date", value=pd.to_datetime("2019-01-01"), key="start_date")
    end_date = st.date_input("End date", value=pd.to_datetime("2019-12-31"), key="end_date")
    
    # Automatically display the asset price graph based on the selected dates
    buf = generate_asset_price_graph(asset_name, start_date, end_date)
    st.image(buf, use_column_width=True)

# Re-Scale Graphic tab
with tab2:
    st.header("Re-Scale Graphic")

    # Input for date range
    start_date = st.date_input("Start date", value=pd.to_datetime("2019-01-01"), key="rescale_start_date")
    end_date = st.date_input("End date", value=pd.to_datetime("2019-12-31"), key="rescale_end_date")
    
    # Automatically display the normalized graph based on the selected date range
    buf = generate_normalized_graph(start_date, end_date)
    st.image(buf, use_column_width=True)

# Returns tab
with tab3:
    st.header("Returns")
    st.write("Content for Returns tab")

# Correlation tab
with tab4:
    st.header("Correlation")
    
    # Automatically display the correlation matrix
    buf = generate_correlation_matrix()
    st.image(buf, use_column_width=True)
    
    st.write("""
    <div class="correlation-text">
    Correlation between S&P 500 PRICE IN USD and BITCOIN PRICE IN USD is very strong (0.861), and is significant (p-value: 0.0000).<br><br>
    Correlation between S&P 500 PRICE IN USD and ETHEREUM PRICE IN USD is very strong (0.889), and is significant (p-value: 0.0000).<br><br>
    Correlation between GOLD PRICE IN USD and BITCOIN PRICE IN USD is strong (0.622), and is significant (p-value: 0.0000).<br><br>
    Correlation between GOLD PRICE IN USD and ETHEREUM PRICE IN USD is moderate (0.595), and is significant (p-value: 0.0000).<br><br>
    Correlation between BITCOIN PRICE IN USD and ETHEREUM PRICE IN USD is very strong (0.923), and is significant (p-value: 0.0000).
    </div>
    """, unsafe_allow_html=True)

# Regression tab
with tab5:
    st.header("Regression")
    
    # Asset selection
    asset_name = st.selectbox("Select an asset for regression", ["S&P 500 PRICE IN USD", "GOLD PRICE IN USD", "BITCOIN PRICE IN USD", "ETHEREUM PRICE IN USD"], key="regression_asset")

    # Buttons for Linear Regression and Log-Linear Regression
    regression_type = st.radio("Select Regression Type", ["Linear Regression", "Log-Linear Regression"])

    # Placeholder content based on the selected regression type
    if regression_type:
        st.write(f"You have selected {regression_type.replace('-', ' ').title()} for {asset_name}.")
        # Add logic to display the regression analysis results based on `regression_type` and `asset_name`

# ARIMA tab
with tab6:
    st.header("ARIMA")
    
    # Asset selection
    asset_name = st.selectbox("Select an asset for ARIMA model", ["Bitcoin", "Ethereum"], key="arima_asset")
    
    # Placeholder content for ARIMA model
    st.write(f"ARIMA model for {asset_name}.")
    # Add logic to display ARIMA model results based on `asset_name`

# LSTM 1 tab
with tab7:
    st.header("LSTM 1")
    
    # Asset selection
    asset_name = st.selectbox("Select an asset for LSTM 1 model", ["Bitcoin", "Ethereum"], key="lstm1_asset")
    
    # Placeholder content for LSTM 1 model
    st.write(f"LSTM 1 model for {asset_name}.")
    # Add logic to display LSTM 1 model results based on `asset_name`

# LSTM 2 tab
with tab8:
    st.header("LSTM 2")
    
    # Asset selection
    asset_name = st.selectbox("Select an asset for LSTM 2 model", ["Bitcoin", "Ethereum"], key="lstm2_asset")
    
    # Placeholder content for LSTM 2 model
    st.write(f"LSTM 2 model for {asset_name}.")
    # Add logic to display LSTM 2 model results based on `asset_name`
