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

* All stock market sectors but Bitcoin are procyclical: as the GDP growth increases, so do stock market prices. However, some sectors more than others: when the economy is thriving, people are more confident on consumer discretionary, tech and healthcare stocks, and they are not as confident on investing in financials and utilities stocks.
* There is a positive relationship between the CPI and stock market prices. This can signify that as inflation increases, stocks generally act as a hedge against inflation, thus people decide to invest in the stock market rather than keeping assets in savings
* There is a negative relationship between the euro/dollar exchange rate and stock prices. This could be due to the fact that if the currency is stronger, people value current earnings more than future earnings
* There is a weak positive relationship between the price of gold and the price of stocks, signifying syncronised growth
* Bitcoin values are different from other stocks because the regressors are not as statistically significant due to the smaller sample size (Bitcoin data starts in 2014). The coefficients can still be interpretable: the fact that Bitcoin is acyclical can signify that people see crypto as a safe haven during economic downturns, as it happened in 2008 and during the COVID-19 pandemic. A strong positive relationship can signify that crypto is seen as a store of value during inflationary pressures

To check for the model's reliability and multicollinearity, I ran some tests in R. To check for the model's reliability, I used the stepwise() function:

  ```{r stepwise}
  stepwise_model <- step(lm(avg_technology ~ log(gdp_growth_rate) + interest_rate_us + cpi + 
                          euro_dollar + avg_vix_close + avg_gold_us_price, 
                          data = Data_Quarterly_MT), direction = "both")
  summary(stepwise_model)
  ```


To check for multicollinearity, I used the vif() function:

  ```{r multicollinearity}
  vif(stepwise_model)
  ```

I then compared different models from their error terms and AIC values to determine which one fit the data better:

  ```{r comparisons}
  # Extract R-squared and Adjusted R-squared: simple vs stepwise
  cat("\nTechreg_s.lm R-squared:", summary(Techreg_s.lm)$r.squared, 
    "Adjusted R-squared:", summary(Techreg_s.lm)$adj.r.squared)

  cat("\nstepwise_model R-squared:", summary(stepwise_model)$r.squared, 
    "Adjusted R-squared:", summary(stepwise_model)$adj.r.squared)

  # Extract AIC and BIC
  cat("\nTechreg_s.lm AIC:", AIC(Techreg_s.lm), "BIC:", BIC(Techreg_s.lm))
  cat("\nstepwise_model AIC:", AIC(stepwise_model), "BIC:", BIC(stepwise_model)) 
  ```

## Part 2: Forecasting stock market prices with ARIMA models

