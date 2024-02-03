from __future__ import (absolute_import, division, print_function,unicode_literals)
import streamlit as st
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
from io import StringIO
import contextlib
import textwrap
import backtrader as bt
import sys  # Library for system-specific parameters and functions


st.set_page_config(layout="wide")



# Set page title
st.title("QuantQuip")

# Function to fetch real-time stock data during market hours
def fetch_realtime_stock_data(ticker_symbol, period, interval):
    current_time = datetime.now().time()
    
    try:
            stock_data = yf.download(ticker_symbol, period=period, interval=interval)
    except Exception as e:
            # Retry with a different interval if the initial request fails
            if "15m data not available" in str(e):
                st.warning(f"15-minute data not available for the specified period. Fetching hourly data instead.")
                stock_data = yf.download(ticker_symbol, period=period, interval="1h")
            else:
                raise
    return stock_data
    

# Default values for period and interval
default_period = "1d"
default_interval = "1m"

# Fetch and display real-time stock data for NSE
nse_ticker_symbol = "^NSEI"
nse_stock_data = fetch_realtime_stock_data(nse_ticker_symbol, default_period, default_interval)

# Fetch and display real-time stock data for Sensex
sensex_ticker_symbol = "^BSESN"
sensex_stock_data = fetch_realtime_stock_data(sensex_ticker_symbol, default_period, default_interval)

# Calculate the percentage change for NSE
nse_percentage_change = (nse_stock_data['Close'].iloc[-1] - nse_stock_data['Close'].iloc[0]) / nse_stock_data['Close'].iloc[0] * 100

# Calculate the percentage change for Sensex
sensex_percentage_change = (sensex_stock_data['Close'].iloc[-1] - sensex_stock_data['Close'].iloc[0]) / sensex_stock_data['Close'].iloc[0] * 100

# Determine overall market condition
overall_market_condition = 'Bullish' if nse_percentage_change > 0 and sensex_percentage_change > 0 else 'Bearish'

# Sidebar for navigation
# Add a title and a selectbox to the sidebar
st.sidebar.markdown("## **Navigation**")
page_selection = st.sidebar.selectbox("", ["Home", "Backtesting", "About Us"])

# Add some space between the selectbox and the next item
st.sidebar.markdown("---")

# Continue with the rest of your sidebar code...


# Default page
def plot_chart(stock_data, title, subheader):
    fig = px.line(stock_data, x=stock_data.index, y="Close", title=title)
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Closing Price')
    st.subheader(subheader)
    if overall_market_condition == 'Bullish':
        fig.update_traces(line_color='green')
    else:
        fig.update_traces(line_color='red')
    st.plotly_chart(fig, use_container_width=True, width=1200)
    st.subheader(f"Latest Data for {title}")
    st.write(stock_data.tail())  # Display the last rows of the DataFrame

if page_selection == "Home":
    # Display overall market condition and real-time stock charts
    st.subheader(f"Overall Market Condition: {overall_market_condition}")

    # Display NSE and Sensex charts side by side with larger width
    if nse_stock_data is not None and sensex_stock_data is not None:
        col1, col2 = st.columns(2)

        with col1:
            plot_chart(nse_stock_data, "Real-time Nifty Chart", "Real-time Nifty Chart:")

        with col2:
            plot_chart(sensex_stock_data, "Real-time Sensex Chart", "Real-time Sensex Chart:")
    else:
        st.warning("Nifty or Sensex market is closed. Real-time data is available only during market hours.")
    st.write("Auto-refreshing every 1 minute.")
























elif page_selection == "Backtesting":


    # Function to run backtest (replace with your actual backtest function)
    def run_backtest(strategy, cash, data):
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy)
        cerebro.broker.setcash(cash)
        cerebro.adddata(data)
        cerebro.run()

        # Streamlit code
        st.title('Backtrader Backtest App')

    # Text area for user to enter Python code
    strategy_code = st.text_area('Enter your Python code:', value='', height=None, max_chars=None, key=None)

    # Input for initial cash
    cash = st.number_input('Enter initial cash:', value=10000.0)
    if (cash < 0.0):
        st.warning('Initial cash must be a positive number')
        

    # Input for data file
    data_file = st.text_input('Enter path to data file:', value='')

    # Button to run backtest
    if st.button('Run Backtest'):
        # Redirect stdout to a string buffer
        buf = StringIO()
        sys.stdout = buf

        # Wrap code in a function and add necessary imports
        code = textwrap.dedent(f"""
        import backtrader as bt
        import pandas as pd

        {strategy_code}

        # Load data
        data = bt.feeds.PandasData(dataname=pd.read_csv('{data_file}'))

        # Run backtest
        run_backtest(MyStrategy, {cash}, data)
        """)

        # Execute the code
        exec(code)

        # Get stdout contents
        output = buf.getvalue()

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Display output
        st.text(output)


































elif page_selection == "About Us":
    st.title("About Us")
    st.markdown("""
    **Welcome to our stock analysis and backtesting platform.** 
    -  We are a group of passionate undergraduate students who are on a mission to make life easier for people dipping their toes into the fintech world. 
    - Our platform provides users with the tools they need to backtest genetic algorithms and carry out successful trades. 
    - With our innovative approach, we aim to revolutionize the **way people interact with financial markets** and **empower them to make informed decisions**.
    """)

    st.subheader("Our Team")
    st.markdown("""
    Meet our team of talented individuals who are dedicated to making a difference in the fintech world.
    """)

    col1, col2, col3 ,col4 = st.columns(4)
    with col1:
        st.image("https://avatars.githubusercontent.com/u/45279662?v=4", use_column_width=True)
        st.markdown("""
        #### Krishnatejaswi S
        Python Developer
        """)
    with col2:
        st.image("https://avatars.githubusercontent.com/u/45279662?v=4", use_column_width=True)
        st.markdown("""
        #### Vinayak C
        ML Engineer
        """)
    with col3:
        st.image("https://avatars.githubusercontent.com/u/45279662?v=4", use_column_width=True)
        st.markdown("""
        #### Bipin Raj C
        Python Developer
        """)
    with col4:
        st.image("https://avatars.githubusercontent.com/u/45279662?v=4", use_column_width=True)
        st.markdown("""
        #### Ananya Bhat
        Python Developer
        """)




