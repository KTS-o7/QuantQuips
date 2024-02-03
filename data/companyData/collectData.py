# Import necessary libraries
import yfinance as yf  # Library for downloading historical market data from Yahoo Finance
import sys  # Library for system-specific parameters and functions
import os  # Library for interacting with the operating system
import pandas as pd  # Library for data manipulation and analysis

# Define a function to download and save stock data
def download_data(company_name,country):
    # Use the Ticker module in yfinance to download the company's stock data
    stock = yf.Ticker(company_name)
    # Get the historical market data, here we are getting data from the start of 2000 till the end of 2024
    data = stock.history(start="2000-01-01", end="2024-12-31")

    # Check if the downloaded data is empty (this could happen if the company name is not found)
    if data.empty:
        print(f"No data found for {company_name}")
        return  # Return from the function if no data is found

    # If data is found, save it to a csv file with the company's name
    data.to_csv(f'./{country}/{company_name}.csv')  # Save the data in a CSV file in a directory named after the country

# Use the function to download data
if __name__ == "__main__":
    # Get the country code from the command line arguments
    companies = sys.argv[1]
    # Depending on the country code, set the list of companies to download data for
    if companies == "IND":
        companieslist="indian_companies"
    elif companies == "US":
        companieslist="us_companies"
    # Read the CSV file with ticker symbols
    ticker_symbols = pd.read_csv(f'../TickerList/{companieslist}.csv')

    # Limit to the first 2 companies
    ticker_symbols = ticker_symbols.head(2)

    # Loop through each ticker symbol and download the data
    for ticker in ticker_symbols['Ticker']:
        download_data(ticker,companies)  # Download data for each ticker
