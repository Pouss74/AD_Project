import streamlit as st
from datetime import datetime

# Import the backend functions
from backend import generate_asset_price_graph, generate_normalized_graph, generate_correlation_matrix

# Application title
st.title('Asset Price Visualization Application')

# Asset selection
asset_name = st.selectbox('Select an asset',
                          options=['S&P 500 PRICE IN USD', 'GOLD PRICE IN USD', 'BITCOIN PRICE IN USD',
                                   'ETHEREUM PRICE IN USD'])

# Start and end date selection
start_date = st.date_input('Start date', value=datetime(2019, 1, 1))
end_date = st.date_input('End date', value=datetime(2020, 1, 1))

# Button to generate the individual asset plot
if st.button('Show asset plot'):
    try:
        # Call the backend function to generate the individual asset plot
        buf = generate_asset_price_graph(asset_name, start_date, end_date)

        # Display the individual asset plot
        st.image(buf, use_column_width=True)

    except ValueError as e:
        st.error(e)

# Button to generate the normalized assets plot
if st.button('Show normalized assets plot'):
    try:
        # Call the backend function to generate the normalized assets plot
        buf = generate_normalized_graph(start_date, end_date)

        # Display the normalized assets plot
        st.image(buf, use_column_width=True)

    except ValueError as e:
        st.error(e)

# Button to generate the correlation matrix plot
if st.button('Show correlation matrix'):
    try:
        # Call the backend function to generate the correlation matrix plot
        buf = generate_correlation_matrix()

        # Display the correlation matrix plot
        st.image(buf, use_column_width=True)

    except ValueError as e:
        st.error(e)
