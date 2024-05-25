INTERCONNECTIVITY BETWEEN DIGITAL AND TRADITIONAL ASSET PRICES

1. INTRODUCTION

This project aims to explore the interconnectivity between digital and traditional asset prices using various data
analysis and machine learning techniques. Specifically, the project focuses on studying the correlation between the
S&P 500, gold, Bitcoin, and Ethereum over the past five years. The core question addressed is whether the price
fluctuations of Bitcoin and Ethereum have been significantly correlated with those of gold and major stock indices.
The analysis begins with an exploration of the data, examining returns, correlations, and performing linear
regression. Following this, both LSTM (Long Short-Term Memory) models and ARIMA (AutoRegressive
Integrated Moving Average) models are employed to provide deeper insights for Bitcoin and Ethereum price
predictions. The analysis is conducted using Python in a Jupyter Notebook environment.

2. REQUIREMENTS

To run this project, you need Python 3.9.12 and the following packages installed in their latest accessible versions:

• Pandas - version 2.2.2
• Matplotlib - version 3.8.4
• Seaborn - version 0.13.2
• Numpy - version 1.26.4
• Warnings (built-in module)
• TensorFlow - version 2.8.0
• Scipy - version 1.13.0
• Scikit-learn - version 1.4.2
• Statsmodels - version 0.14.2

To run the project, clone the GitHub repository to your local machine and navigate to the project directory. Ensure
you have the CSV file of the database named DataCapstone.csv in the same directory as the Jupyter Notebook.
The CSV file should use a semicolon (;) as the delimiter and commas (,) for decimal separation. Open the Jupyter
Notebook and run the cells to execute the analysis.
If you have any questions for running the code, please ask me at maxime.poussard@unil.ch.

3. STREAMLIT INTERFACE

The Streamlit application provides an interactive platform for analyzing the interconnectivity between digital assets (Bitcoin and Ethereum) and traditional financial assets (Gold and S&P 500). Users can interact with data, visualize trends, and forecast asset prices through multiple predictive models, including Linear Regression, ARIMA, and LSTM. Due to their computational intensity, ARIMA and LSTM predictions are displayed as precomputed images, ensuring quick and seamless user experience. The application allows for individual and comparative analysis of asset prices and returns, supported by a correlation matrix to show relationships between different assets. Users can customize their analysis by selecting different asset models, date ranges, and predictive models. The interface displays individual and comparative price graphs, correlation matrix, returns, and correlations between asset returns, providing clear and interactive visualizations to aid in data-driven decision-making.

To better understand the development of the website, please refer to the "app" and "backend" codes.

Website : https://programmation-capstone.streamlit.app
