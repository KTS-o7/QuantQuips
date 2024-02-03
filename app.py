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
    </style>
    """,
    unsafe_allow_html=True
)

# Set page title
st.title("Real-time Nifty Chart")

# Function to fetch real-time Nifty data during market hours
def fetch_realtime_nifty_data(ticker_symbol, period, interval):
    current_time = datetime.now().time()
    if current_time >= datetime.strptime("09:15", "%H:%M").time() and current_time <= datetime.strptime("15:30", "%H:%M").time():
        nifty_data = yf.download(ticker_symbol, period=period, interval=interval)
        return nifty_data
    else:
        return None

# Sidebar for user input
st.sidebar.header("User Input")

# Get user input
ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., ^NSEI for Nifty):", "^NSEI")
period = st.sidebar.selectbox("Select Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
interval = st.sidebar.selectbox("Select Interval:", ["1m", "2m", "5m", "15m", "30m", "1h", "1d"])

# Fetch real-time Nifty data
nifty_data = fetch_realtime_nifty_data(ticker_symbol, period, interval)

# Display real-time Nifty chart
if nifty_data is not None:
    fig = px.line(nifty_data, x=nifty_data.index, y="Close", title=f"Real-time {ticker_symbol} Chart")
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Closing Price')

    st.subheader(f"Real-time {ticker_symbol} Chart:")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Market is closed. Real-time data is available only during market hours.")

# Auto-refresh every 1 minute
st.text("Auto-refreshing every 1 minute")
st.text(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Run the app

