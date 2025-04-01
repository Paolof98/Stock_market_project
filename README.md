# Analysing and forecasting stock market returns: project for the Data Analystics bootcamp at the Curious Lounge Academy

* Part 1: Analysing the stock market: regression models
* Part 2: Forecasting stock market returns with ARIMA models
* Part 3: Beyond ARIMA models: looking into Machine Learning models

To do this project I analysed economic data from the US, from the end of the 1990s to the end of 2024. 

The independent variables I used were:

* Real GDP, real GDP gorwth rate
* CPI
* Interest rate
* Unemployment rate
* Eur/Dollar exchange rate
* VIX: market volatility
* Gold price

The sectors of the tock market I looked into were:

* Technology
* Healthcare
* Financials
* Coonsumer discretionary
* Consumer staples
* Industrials
* Utilities
* S&P 500
* Bitcoin, as a proxy for crypto


## Part 1: Analysing the stock market: regression models

The regression model I used to analyse relationships between the stock market and economic variables was:

![](regression%20model.jpg)

We can interpret the regression coefficients with the following:

* A positive relationship between the log of the GDP growth rate and the specified sector of the stock market (beta 1 > 0) means that the stock market sector is procyclical: as the GDP growth increases, so does the price of the stock market indices in that sector. A negative relationship between the log of the GDP growth rate and the specified sector of the stock market (beta 1 < 0) means that the stock market sector is anticyclical: as the GDP growth increases, the price of the stock market indices in that sector decreases. The stock market sector is acyclical if beta 1 = 0
* A positive relationship between the CPI and the stock market price (beta 2 > 0) can signify that stocks act as a hedge against inflation, a negative relationship (beta 2 <0) can highlight the fact that expected future earnings are less desirable, so stock prices decrease
* A negative relationship between the euro/dollar exchange rate and the stock market price (beta 3 < 0) can illustrate a substitution effect between euros and stock indices
* A positive relationship between the price of gold and the price of stocks (beta 4 > 0) can signify that there is syncronised growth between gold and stock indices, which could be due to inflationary pressures, a negative relationship (beta 4 > 0) could demonstrate a substitution effect between stocks and gold, as gold could be considered a safe asset. If stock prices fall, then gold could be seen as a store of value

The findings are summarised in the following table:

![](Regression%20models%20results.jpg)

We can interpret results with the following:

* 
