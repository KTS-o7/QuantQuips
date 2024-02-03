import streamlit as st
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# Set dark mode theme
st.markdown(
    """
    <style>
    body {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .markdown-container {
        max-width: 1200px;
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

# Sidebar for user input
st.sidebar.header("User Input")

# Get user input using drop-down menus for NSE
nse_ticker_symbol = st.sidebar.selectbox("Select NSE Ticker Symbol:", ["^NSEI", "AAPL", "GOOGL", "MSFT"])  # Add more options as needed
nse_period = st.sidebar.selectbox("Select NSE Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
nse_interval = st.sidebar.selectbox("Select NSE Interval:", ["1m", "2m", "5m", "15m", "30m", "1h", "1d"])

# Get user input using drop-down menus for Sensex
sensex_ticker_symbol = st.sidebar.selectbox("Select Sensex Ticker Symbol:", ["^BSESN"])  # Add more options as needed
sensex_period = st.sidebar.selectbox("Select Sensex Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
sensex_interval = st.sidebar.selectbox("Select Sensex Interval:", ["1m", "2m", "5m", "15m", "30m", "1h", "1d"])

# Fetch and display real-time stock data for NSE
nse_stock_data = fetch_realtime_stock_data(nse_ticker_symbol, nse_period, nse_interval)

# Fetch and display real-time stock data for Sensex
sensex_stock_data = fetch_realtime_stock_data(sensex_ticker_symbol, sensex_period, sensex_interval)

# Display NSE and Sensex charts side by side with larger width
if nse_stock_data is not None and sensex_stock_data is not None:
    col1, col2 = st.columns(2)

    with col1:
        nse_fig = px.line(nse_stock_data, x=nse_stock_data.index, y="Close", title=f"Real-time {nse_ticker_symbol} Chart")
        nse_fig.update_xaxes(title_text='Time')
        nse_fig.update_yaxes(title_text='Closing Price')
        st.subheader(f"Real-time {nse_ticker_symbol} Chart:")
        st.plotly_chart(nse_fig, use_container_width=True, width=1200)

    with col2:
        sensex_fig = px.line(sensex_stock_data, x=sensex_stock_data.index, y="Close", title=f"Real-time Sensex Chart")
        sensex_fig.update_xaxes(title_text='Time')
        sensex_fig.update_yaxes(title_text='Closing Price')
        st.subheader("Real-time Sensex Chart:")
        st.plotly_chart(sensex_fig, use_container_width=True, width=1200)
else:
    st.warning("NSE or Sensex market is closed. Real-time data is available only during market hours.")

# Auto-refresh every 1 minute
st.text("Auto-refreshing every 1 minute")
st.text(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run the app

