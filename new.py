import streamlit as st
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
# Set dark mode theme
st.markdown(
    """
    <style>
    body {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .markdown-container {
        max-width: 2000px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page title
st.title("Real-time Stock Charts")

# Function to fetch real-time stock data during market hours
def fetch_realtime_stock_data(ticker_symbol, period, interval):
    current_time = datetime.now().time()
    if current_time >= datetime.strptime("09:15", "%H:%M").time() and current_time <= datetime.strptime("15:30", "%H:%M").time():
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
    else:
        return None

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
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Home", "Backtesting", "About Us"])

# Default page
if page_selection == "Home":
    # Display overall market condition and real-time stock charts
    st.subheader(f"Overall Market Condition: {overall_market_condition}")

    # Display NSE and Sensex charts side by side with larger width
    if nse_stock_data is not None and sensex_stock_data is not None:
        col1, col2 = st.columns(2)

        with col1:
            nse_fig = px.line(nse_stock_data, x=nse_stock_data.index, y="Close", title=f"Real-time {nse_ticker_symbol} Chart")
            nse_fig.update_xaxes(title_text='Time')
            nse_fig.update_yaxes(title_text='Closing Price')
            st.subheader(f"Real-time {nse_ticker_symbol} Chart:")
            if overall_market_condition == 'Bullish':
                nse_fig.update_traces(line_color='green')
            else:
                nse_fig.update_traces(line_color='red')
            st.plotly_chart(nse_fig, use_container_width=True, width=1200)

            # Table for NSE
            st.subheader(f"Latest Data for {nse_ticker_symbol}")
            st.write(nse_stock_data.tail())  # Display the last rows of the DataFrame

        with col2:
            sensex_fig = px.line(sensex_stock_data, x=sensex_stock_data.index, y="Close", title=f"Real-time Sensex Chart")
            sensex_fig.update_xaxes(title_text='Time')
            sensex_fig.update_yaxes(title_text='Closing Price')
            st.subheader("Real-time Sensex Chart:")
            if overall_market_condition == 'Bullish':
                sensex_fig.update_traces(line_color='green')
            else:
                sensex_fig.update_traces(line_color='red')
            st.plotly_chart(sensex_fig, use_container_width=True, width=1200)

            # Table for Sensex
            st.subheader("Latest Data for Sensex")
            st.write(sensex_stock_data.tail())  # Display the last rows of the DataFrame
    else:
        st.warning("NSE or Sensex market is closed. Real-time data is available only during market hours.")

elif page_selection == "Backtesting":
    # Placeholder for backtesting page
    st.title("Backtesting Page")
    st.write("This is the backtesting page. Add your backtesting content here.")

elif page_selection == "About Us":
    # Placeholder for about us page
    st.title("About Us")
    st.write("Welcome to our stock analysis and backtesting platform. Add information about your team or project here.")

# Auto-refresh every 1 minute
st.write("Auto-refreshing every 1 minute.")

# Run the app

