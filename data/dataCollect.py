# Import necessary libraries
import yfinance as yf  # Library for downloading historical market data from Yahoo Finance
import sys  # Library for system-specific parameters and functions
import os  # Library for interacting with the operating system

# Define a function to download and save stock data
def download_data(company_name):
    # Use the Ticker module in yfinance to download the company's stock data
    stock = yf.Ticker(company_name)
    # Get the historical market data, here we are getting data from the start of 2000 till the end of 2024
    data = stock.history(start="2000-01-01", end="2024-12-31")

    # Check if the downloaded data is empty (this could happen if the company name is not found)
    if data.empty:
        print(f"No data found for {company_name}")
        sys.exit()  # Exit the program if no data is found

    # If data is found, save it to a csv file with the company's name
    data.to_csv(f'{company_name}.csv')

# Use the function to download data
if __name__ == "__main__":
    # Get the company name from the command line arguments, if not provided, get it from the environment variables, if not there, default to 'AAPL'
    company_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv('COMPANY_NAME', 'AAPL')
    # Call the function with the company name
    download_data(company_name)
