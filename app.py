import streamlit as st
import pandas as pd
from backend import generate_asset_price_graph, generate_rescaled_plot, generate_correlation_matrix, load_and_prepare_data, generate_plot, plot_regression

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
    .css-1q8dd3e, .css-1d391kg, .css-18e3th9, .stSelectbox label, .stDateInput label, .stRadio label, .stRadio span {
        color: #ffffff !important; /* White text for labels */
    }
    .correlation-text {
        margin-top: 20px;
        line-height: 1.6;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    .metrics {
        margin-top: 20px;
        line-height: 1.6;
    }
    .description {
        text-align: center;
        color: #ffffff;
        margin-top: 20px;
        font-size: 1.1em;
        line-height: 1.6;
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
st.markdown(
    """
    <div class="disclaimer">
    Disclaimer: These charts are for research purposes only and do not constitute investment advice. Investing involves risks.
    </div>
    """, 
    unsafe_allow_html=True
)
st.title("Future is Yours")

st.markdown(
    """
    <div class="description">
    Managing asset correlations and predicting price movements can be time-consuming and complex. Imagine what you could achieve if you had an efficient, streamlined way to handle these tasks!<br><br>
    My project simplifies the analysis of digital (Bitcoin and Ethereum) and traditional (Gold and S&P 500) asset prices from January 1, 2019, to April 9, 2024. Utilizing statistical analysis and predictive models like regression, ARIMA, and LSTM networks, we uncover the dynamic interactions between these assets. This enables investors, analysts, and policymakers to make informed decisions and optimize their investment and risk management strategies.<br><br>
    Thank you for using our site to enhance your financial insights.
    </div>
    """,
    unsafe_allow_html=True
)

# Add some space before the tabs
st.markdown("<br><br>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Historical Data", "Re-Scale Graphic", "Returns", "Correlation", "Regression", "ARIMA", "LSTM 1", "LSTM 2"])

# Historical Data tab
with tab1:
    st.header("Historical Data")
    
    # Input for asset price graph
    asset_name = st.selectbox("Select an asset", ["S&P 500 PRICE IN USD", "GOLD PRICE IN USD", "BITCOIN PRICE IN USD",
                                                  "ETHEREUM PRICE IN USD"], key="historical_asset")
    start_date = st.date_input("Start date", value=pd.to_datetime("2019-01-01"), key="historical_start_date")
    end_date = st.date_input("End date", value=pd.to_datetime("2019-12-31"), key="historical_end_date")
    
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
    buf = generate_rescaled_plot(start_date, end_date)
    st.image(buf, use_column_width=True)

# Returns tab
with tab3:
    st.header("Returns")
    
    # Load and prepare data
    data = load_and_prepare_data()

    # User inputs for asset selection and date range
    asset = st.selectbox('Select Asset', ['S&P 500', 'GOLD', 'BITCOIN', 'ETHEREUM'], key="returns_asset")
    start_date = st.date_input('Start date', pd.to_datetime('2021-01-01'), key="returns_start_date")
    end_date = st.date_input('End date', pd.to_datetime('2022-01-01'), key="returns_end_date")

    # Generate and display the plot if dates and asset are selected
    if start_date and end_date and asset:
        plot_buf = generate_plot(data, asset, start_date, end_date)
        st.image(plot_buf, caption=f'{asset} Returns from {start_date} to {end_date}')
    
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
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Asset selection
    asset_name = st.selectbox("Select an asset for regression", ["S&P 500", "GOLD", "BITCOIN", "ETHEREUM"], key="regression_asset")

    # Buttons for Linear Regression and Log-Linear Regression
    regression_type = st.radio("Select Regression Type", ["Linear Regression", "Log-Linear Regression"], key="regression_type")
    
    # Date range selection
    start_date = st.date_input('Start date', pd.to_datetime('2019-01-01'), key="regression_start_date")
    end_date = st.date_input('End date', pd.to_datetime('2022-01-01'), key="regression_end_date")

    # Placeholder content based on the selected regression type
    if regression_type and start_date and end_date and asset_name:
        st.write(f"You have selected {regression_type.replace('-', ' ').title()} for {asset_name}.")
        
        # Determine whether to apply log transformation
        log = regression_type == "Log-Linear Regression"
        
        # Generate and display the plot
        plot_buf = plot_regression(data, asset_name, start_date, end_date, log)
        st.image(plot_buf, caption=f'{asset_name} {"Log " if log else ""}Price Evolution with Regression Line from {start_date} to {end_date}')

# ARIMA tab
with tab6:
    st.header("ARIMA")
    
    # Asset selection
    asset_name = st.selectbox("Select an asset for ARIMA model", ["Bitcoin", "Ethereum"], key="arima_asset")
    
    # Automatically display the ARIMA forecast based on the selected asset
    arima_image_path = get_forecast_image_path(asset_name, "ARIMA")
    st.image(arima_image_path, use_column_width=True)

# LSTM 1 tab
with tab7:
    st.header("LSTM 1")
    
    # Asset selection
    asset_name = st.selectbox("Select an asset for LSTM 1 model", ["Bitcoin", "Ethereum"], key="lstm1_asset")
    
    # Automatically display the LSTM 1 forecast based on the selected asset
    lstm1_image_path = get_forecast_image_path(asset_name, "LSTM")
    st.image(lstm1_image_path, use_column_width=True)
    
    if asset_name == "Bitcoin":
        st.write("""
        <div class="metrics">
        Bitcoin Prediction Metrics of the training:<br>
        <br>Mean Absolute Error: 1502.8202759051017<br>
        Mean Absolute Percentage Error: 3.5216085420893433%<br>
        R-squared Score: 0.968159835080644
        </div>
        """, unsafe_allow_html=True)
    elif asset_name == "Ethereum":
        st.write("""
        <div class="metrics">
        Ethereum Prediction Metrics of the training:<br>
        <br>Mean Absolute Error: 104.6619032567737<br>
        Mean Absolute Percentage Error: 4.634681268119115%<br>
        R-squared Score: 0.9399836079679001
        </div>
        """, unsafe_allow_html=True)

# LSTM 2 tab
with tab8:
    st.header("LSTM 2")
    
    # Asset selection
    asset_name = st.selectbox("Select an asset for LSTM 2 model", ["Bitcoin", "Ethereum"], key="lstm2_asset")
    
    # Automatically display the LSTM 2 forecast based on the selected asset
    lstm2_image_path = get_forecast_image_path(asset_name, "LSTM with correlation")
    st.image(lstm2_image_path, use_column_width=True)
    
    if asset_name == "Bitcoin":
        st.write("""
        <div class="metrics">
        Bitcoin Prediction Metrics of the training:<br>
        <br>Mean Absolute Error: 3111.369628773971<br>
        Mean Absolute Percentage Error: 6.991318338500592%<br>
        R-squared Score: 0.884409707322686
        </div>
        """, unsafe_allow_html=True)
    elif asset_name == "Ethereum":
        st.write("""
        <div class="metrics">
        Ethereum Prediction Metrics of the training:<br>
       <br> Mean Absolute Error: 143.032266023599<br>
        Mean Absolute Percentage Error: 6.550255220961676%<br>
        R-squared Score: 0.9114275560790139
        </div>
        """, unsafe_allow_html=True)
