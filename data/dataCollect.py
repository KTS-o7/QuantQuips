import yfinance as yf
import sys
import os

def download_data(company_name):
    # Download stock data 
    stock = yf.Ticker(company_name)
    data = stock.history(start="2000-01-01", end="2024-12-31")

    # Check if data is empty (company not found)
    if data.empty:
        print(f"No data found for {company_name}")
        sys.exit()

    # Save data to csv file
    data.to_csv(f'{company_name}.csv')

# Use the function
if __name__ == "__main__":
    company_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv('COMPANY_NAME', 'AAPL')
    download_data(company_name)
