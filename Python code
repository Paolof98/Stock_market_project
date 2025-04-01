pip install yfinance

# Define sector ETFs (ticker symbols)
sector_etfs = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Consumer Staples": "XLP"

# Download historical data (weekly)
sector_data = {}

for sector, ticker in sector_etfs.items():
    df = yf.download(ticker, start="1980-01-01", interval="1d", auto_adjust=False)  # Disable auto-adjust
    sector_data[sector] = df["Adj Close"]  # Store adjusted closing prices


for sector, ticker in sector_etfs.items():
    print(f"Checking {sector} ({ticker})...")
    df = yf.download(ticker, start="1980-01-01", interval="1d", auto_adjust=False)
    print(df.head())  # Print first few rows of data
    print(df.columns)  # Print available column names
    break  # Only check one sector for now




# Dictionary to store sector data
sector_data = {}

for sector, ticker in sector_etfs.items():
    print(f"Downloading data for {sector} ({ticker})...")
    
    df = yf.download(ticker, start="1980-01-01", interval="1d", auto_adjust=False)
    
    # Access 'Adj Close' using MultiIndex
    if ("Adj Close", ticker) in df.columns:
        sector_data[sector] = df[("Adj Close", ticker)]
    else:
        print(f"⚠️ Warning: 'Adj Close' missing for {sector} ({ticker}). Skipping.")



# Convert to DataFrame if data exists
if sector_data:
    sector_prices = pd.DataFrame(sector_data)
    sector_prices.to_csv("sector_stock_data.csv")
    print("✅ Sector data downloaded and saved successfully!")
else:
    print("❌ No valid sector data found. Please check your downloads.")


# Define the S&P 500 ETF ticker
sp500_ticker = "SPY"

# Download historical data for S&P 500
df_sp500 = yf.download(sp500_ticker, start="1980-01-01", interval="1d", auto_adjust=False)

# Save the data to a CSV file
df_sp500.to_csv("sp500_data.csv")

print("✅ S&P 500 data downloaded and saved successfully!")


import yfinance as yf

# Download daily Bitcoin price data from 2008 to today
btc_data = yf.download("BTC-USD", start="2008-01-01", interval="1d")

# Display the first few rows
print(btc_data.head())

# Save to CSV
btc_data.to_csv("bitcoin_data.csv")



# Download daily Bitcoin price data from 2008 to today
btc_data = yf.download("BTC-USD", start="2010-01-01", interval="1d")

# Display the first few rows
print(btc_data.head())

