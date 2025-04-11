### DataCamp Project: forecasting models

library(tidyverse)
library(dplyr)
library(readr)
library(sf)
library(ggplot2)
library(fpp2) # package from the "Forecasting in R" DataCamp course
library(glmnet)
library(caret)
library(randomForest)
library(ranger)
library(writexl)
library(gridExtra)
library(car)


# PART 1: Load in data

## 1) Daily market data
Daily_Market_Data <- readr::read_csv("C:/Users/Paolo/Desktop/Project/Data from SQL/daily_market_data.csv")


## 2) Monthly frequency Long term
Data_Monthly_LT <- readr::read_csv("C:/Users/Paolo/Desktop/Project/Data from SQL/economy_stockmarket_monthly.csv")


## 3) Monthly frequency Medium term
Data_Monthly_MT <- readr::read_csv("C:/Users/Paolo/Desktop/Project/Data from SQL/economy_sm_monthly_mt.csv")


## 4) Quarterly frequency Long term
Data_Quarterly_LT <- readr::read_csv("C:/Users/Paolo/Desktop/Project/Data from SQL/economy_stockmarket_quarterly.csv")


## 5) Quarterly frequency Medium term
Data_Quarterly_MT <- readr::read_csv("C:/Users/Paolo/Desktop/Project/Data from SQL/economy_sm_quarterly_mt.csv")




# Create time series from data

## Daily stock market data
ts_Stockmarket <- ts(Daily_Market_Data[, 2:11], start = c(1998, 12), frequency = 252)


## Monthly frequency Long term
ts_MonthlyLT <- ts(Data_Monthly_LT[, 2:19], start = c(1956, 2), frequency = 12)


## Monthly frequency Medium term
ts_MonthlyMT <- ts(Data_Monthly_MT[, 2:19], start = c(1999, 2), frequency = 12)


## Quarterly frequency Long term
ts_QuarterlyLT <- ts(Data_Quarterly_LT[, 2:19], start = c(1956, 2), frequency = 4)


## Monthly frequency Long term
ts_QuarterlyMT <- ts(Data_Quarterly_MT[, 2:19], start = c(1999, 1), frequency = 4)








# PART 2: Regression Analysis

# Stock market returns = B0 + B1(GDPgrowth) + B2(IRs) + B3(Inflation) + B4(ER). Quarterly data

## Tech
Techreg.lm <- lm(formula = avg_technology ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar, data = Data_Quarterly_MT)

summary(Techreg.lm)

## Financials
Financials.lm <- lm(formula = avg_financials ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar, data = Data_Quarterly_MT)

summary(Financials.lm)


## Healthcare
Healthcare.lm <- lm(formula = avg_healthcare ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar, data = Data_Quarterly_MT)

summary(Healthcare.lm)


## Consumer discretionary
CD.lm <- lm(formula = avg_consumer_discretionary ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar, data = Data_Quarterly_MT)

summary(CD.lm)


## Utilities
Utilities.lm <- lm(formula = avg_utilities ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar, data = Data_Quarterly_MT)

summary(Utilities.lm)


## Industrials
Industrials.lm <- lm(formula = avg_industrials ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar, data = Data_Quarterly_MT)

summary(Industrials.lm)


## Consumer staples
CS.lm <- lm(formula = avg_consumer_staples ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar, data = Data_Quarterly_MT)

summary(CS.lm)


## Gold
Gold.lm <- lm(formula = avg_gold_us_price ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar, data = Data_Quarterly_MT)

summary(Gold.lm)


## S&P500
SandP.lm <- lm(formula = avg_sandp_close ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar, data = Data_Quarterly_MT)

summary(SandP.lm)


## Bitcoin
Bitcoin.lm <- lm(formula = avg_bitcoin_close ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar, data = Data_Quarterly_MT)

summary(Bitcoin.lm)


# Can also generate GLM models, compare AICs, choose model with lowest one




## TRY NEW MODEL: Stock market returns = B0 + B1(GDPgrowth) + B2(IRs) + B3(Inflation) + B4(ER) + B5(VIX). Quarterly data

## Tech
Techreg_5V.lm <- lm(formula = avg_technology ~ log(gdp_growth_rate) + interest_rate_us + cpi + euro_dollar + avg_vix_close, data = Data_Quarterly_MT)

summary(Techreg_5V.lm)

## Financials
Financials_5V.lm <- lm(formula = avg_financials ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar + avg_vix_close, data = Data_Quarterly_MT)

summary(Financials_5V.lm)


## Healthcare
Healthcare_5V.lm <- lm(formula = avg_healthcare ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar + avg_vix_close, data = Data_Quarterly_MT)

summary(Healthcare_5V.lm)


## Consumer discretionary
CD_5V.lm <- lm(formula = avg_consumer_discretionary ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar + avg_vix_close, data = Data_Quarterly_MT)

summary(CD_5V.lm)


## Utilities
Utilities_5V.lm <- lm(formula = avg_utilities ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar + avg_vix_close, data = Data_Quarterly_MT)

summary(Utilities_5V.lm)


## Industrials
Industrials_5V.lm <- lm(formula = avg_industrials ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar + avg_vix_close, data = Data_Quarterly_MT)

summary(Industrials_5V.lm)


## Consumer staples
CS_5V.lm <- lm(formula = avg_consumer_staples ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar + avg_vix_close, data = Data_Quarterly_MT)

summary(CS_5V.lm)


## Gold
Gold_5V.lm <- lm(formula = avg_gold_us_price ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar + avg_vix_close, data = Data_Quarterly_MT)

summary(Gold_5V.lm)


## S&P500
SandP_5V.lm <- lm(formula = avg_sandp_close ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar + avg_vix_close, data = Data_Quarterly_MT)

summary(SandP_5V.lm)


## Bitcoin
Bitcoin_5V.lm <- lm(formula = avg_bitcoin_close ~ gdp_growth_rate + interest_rate_us + cpi + euro_dollar + avg_vix_close, data = Data_Quarterly_MT)

summary(Bitcoin_5V.lm)




# Extract R-squared and Adjusted R-squared
cat("\nTechreg.lm R-squared:", summary(Techreg.lm)$r.squared, 
    "Adjusted R-squared:", summary(Techreg.lm)$adj.r.squared)

cat("\nTechreg_5V.lm R-squared:", summary(Techreg_5V.lm)$r.squared, 
    "Adjusted R-squared:", summary(Techreg_5V.lm)$adj.r.squared)

# Extract AIC and BIC
cat("\nTechreg.lm AIC:", AIC(Techreg.lm), "BIC:", BIC(Techreg.lm))
cat("\nTechreg_5V.lm AIC:", AIC(Techreg_5V.lm), "BIC:", BIC(Techreg_5V.lm)) # 5 variables better R squared


vif(Techreg_5V.lm)



### TRY SIMPLER MODEL:
# stock returns = B0 + B1(GDPgrowth) + B2(IRs) + B3(CPI)
## Tech
Techreg_s.lm <- lm(formula = avg_technology ~ log(gdp_growth_rate) + interest_rate_us + cpi, data = Data_Quarterly_MT)

summary(Techreg_s.lm)









stepwise_model <- step(lm(avg_technology ~ log(gdp_growth_rate) + interest_rate_us + cpi + 
                          euro_dollar + avg_vix_close + avg_gold_us_price, 
                          data = Data_Quarterly_MT), direction = "both")
summary(stepwise_model)


vif(stepwise_model)



stepwise_model_Nog <- step(lm(avg_technology ~ log(gdp_growth_rate) + interest_rate_us + cpi + 
                          euro_dollar + avg_vix_close, 
                          data = Data_Quarterly_MT), direction = "both")
summary(stepwise_model_Nog)


vif(stepwise_model_Nog)




# Extract R-squared and Adjusted R-squared: simple vs stepwise
cat("\nTechreg_s.lm R-squared:", summary(Techreg_s.lm)$r.squared, 
    "Adjusted R-squared:", summary(Techreg_s.lm)$adj.r.squared)

cat("\nstepwise_model R-squared:", summary(stepwise_model)$r.squared, 
    "Adjusted R-squared:", summary(stepwise_model)$adj.r.squared)

# Extract AIC and BIC
cat("\nTechreg_s.lm AIC:", AIC(Techreg_s.lm), "BIC:", BIC(Techreg_s.lm))
cat("\nstepwise_model AIC:", AIC(stepwise_model), "BIC:", BIC(stepwise_model)) 


# HENCE MODEL IS:
# stock returns = B0 + B1log(gdp growth rate) + B3(CPI) + B4(ER) + B5(Gold price)

## Tech
Techreg_SW.lm <- lm(formula = avg_technology ~ log(gdp_growth_rate) + cpi + euro_dollar + avg_gold_us_price, data = Data_Quarterly_MT)

summary(Techreg_SW.lm)

## Financials
Financials_SW.lm <- lm(formula = avg_financials ~ log(gdp_growth_rate) + cpi + euro_dollar + avg_gold_us_price, data = Data_Quarterly_MT)

summary(Financials_SW.lm)


## Healthcare
Healthcare_SW.lm <- lm(formula = avg_healthcare ~ log(gdp_growth_rate) + cpi + euro_dollar + avg_gold_us_price, data = Data_Quarterly_MT)

summary(Healthcare_SW.lm)


## Consumer discretionary
CD_SW.lm <- lm(formula = avg_consumer_discretionary ~ log(gdp_growth_rate) + cpi + euro_dollar + avg_gold_us_price, data = Data_Quarterly_MT)

summary(CD_SW.lm)


## Utilities
Utilities_SW.lm <- lm(formula = avg_utilities ~ log(gdp_growth_rate) + cpi + euro_dollar + avg_gold_us_price, data = Data_Quarterly_MT)

summary(Utilities_SW.lm)


## Industrials
Industrials_SW.lm <- lm(formula = avg_industrials ~ log(gdp_growth_rate) + cpi + euro_dollar + avg_gold_us_price, data = Data_Quarterly_MT)

summary(Industrials_SW.lm)


## Consumer staples
CS_SW.lm <- lm(formula = avg_consumer_staples ~ log(gdp_growth_rate) + cpi + euro_dollar + avg_gold_us_price, data = Data_Quarterly_MT)

summary(CS_SW.lm)


## S&P500
SandP_SW.lm <- lm(formula = avg_sandp_close ~ log(gdp_growth_rate) + cpi + euro_dollar + avg_gold_us_price, data = Data_Quarterly_MT)

summary(SandP_SW.lm)


## Bitcoin
Bitcoin_SW.lm <- lm(formula = avg_bitcoin_close ~ log(gdp_growth_rate) + cpi + euro_dollar + avg_gold_us_price, data = Data_Quarterly_MT)

summary(Bitcoin_SW.lm)































cor(Data_Quarterly_MT[, c("interest_rate_us", "gdp_growth_rate", "cpi", "euro_dollar", "avg_gold_us_price", "avg_vix_close")])



ggplot(Data_Quarterly_MT, aes(x = interest_rate_us, y = avg_technology)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), col = "blue") +
  labs(title = "Non-Linear Relationship between Interest Rates and Stocks")



lm_nonlinear <- lm(avg_technology ~ interest_rate_us + I(interest_rate_us^2) + cpi + euro_dollar, data = Data_Quarterly_MT)
summary(lm_nonlinear)




library(lmtest)
granger_test <- grangertest(avg_technology ~ interest_rate_us, order = 1, data = Data_Quarterly_MT)
granger_test





















# PART 3: Autoplots

## Daily stock market data
autoplot(ts_Stockmarket, facets = TRUE)

## Monthly frequency Long term
autoplot(ts_MonthlyLT, facets = TRUE)

## Monthly frequency Medium term
autoplot(ts_MonthlyMT, facets = TRUE)

## Quarterly frequency Long term
autoplot(ts_QuarterlyLT, facets = TRUE) 

## Quarterly frequency Medium term. 
autoplot(ts_QuarterlyMT, facets = TRUE) # GDP=1, CPI=2, unemployment rate=3, IR=4, GDPgrowth=5, ER=6, quarterdate=7, tech=8, financials=9, healthcare=10, consumer discretionary=11 etc







