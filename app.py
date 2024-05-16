import streamlit as st
from datetime import datetime

# Debugging imports
try:
    from backend import generate_asset_price_graph
except ImportError as e:
    st.error(f"ImportError: {e}")

# Application title
st.title('Asset Price Visualization Application')

# Asset selection
asset_name = st.selectbox('Select an asset',
                          options=['S&P 500 PRICE IN USD', 'GOLD PRICE IN USD', 'BITCOIN PRICE IN USD',
                                   'ETHEREUM PRICE IN USD'])

# Start and end date selection
start_date = st.date_input('Start date', value=datetime(2019, 1, 1))
end_date = st.date_input('End date', value=datetime(2020, 1, 1))

# Button to generate the plot
if st.button('Show plot'):
    try:
        # Call the backend function to generate the plot
        buf = generate_asset_price_graph(asset_name, start_date, end_date)

        # Display the plot
        st.image(buf, use_column_width=True)

    except ValueError as e:
        st.error(e)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
