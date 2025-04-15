# 1) MODEL WITH MANY VARIABLES, NO HARMONIC SERIES
## Regressors, no harmonic
Xreg_QMT <- cbind(GDPGrowth = ts_QuarterlyMT[, "gdp_growth_rate"], 
			CPI = ts_QuarterlyMT[, "cpi"],
			InterestRate = ts_QuarterlyMT[, "interest_rate_us"],
			GoldPrice = ts_QuarterlyMT[, "avg_gold_us_price"],
			VIX = ts_QuarterlyMT[, "avg_vix_close"],
			ExchangeRate = ts_QuarterlyMT[, "euro_dollar"])

## Fit model, no harmonic
Fit_QMT <- auto.arima(ts_QuarterlyMT[, "avg_technology"], xreg = Xreg_QMT)



## Future values with LT data: fit trend models for each macroeconomic variable
gdp_trend <- auto.arima(ts_QuarterlyLT[, "gdp_growth_rate"])     # GDP growth rate
cpi_trend <- auto.arima(ts_QuarterlyLT[, "cpi"])                 # CPI
ir_trend  <- auto.arima(ts_QuarterlyLT[, "interest_rate_us"])    # IR
gold_trend <- auto.arima(ts_QuarterlyMT[, "avg_gold_us_price"])  # Gold
vix_trend <- auto.arima(ts_QuarterlyMT[, "avg_vix_close"])       # VIX
eurusd_trend <- auto.arima(ts_QuarterlyMT[, "euro_dollar"])      # euro/dollar


### Forecast the next `h` quarters 
future_gdp <- forecast(gdp_trend, h = h)$mean
future_cpi <- forecast(cpi_trend, h = h)$mean
future_ir  <- forecast(ir_trend, h = h)$mean
future_gold <- forecast(gold_trend, h = h)$mean
future_vix <- forecast(vix_trend, h = h)$mean
future_er <- forecast(eurusd_trend, h = h)$mean


### Combine into future matrix
future_macro_trend_LT <- cbind(future_gdp, future_cpi, future_ir, future_gold, future_vix, future_er)


### Now forecast again
forecast(Fit_QMT, xreg = future_macro_trend_LT) %>% autoplot() + ylab("Technology stocks") # this gave better forecast so use future_macro_LT


summary(Fit_QMT)












# Define function to forecast for each sector and collect plots
forecast_sector <- function(sector) {
  
  ## Regressors, no harmonic
  Xreg_QMT_3Variables <- cbind(GDPGrowth = ts_QuarterlyMT[, "gdp_growth_rate"], 
                               CPI = ts_QuarterlyMT[, "cpi"],
                               InterestRate = ts_QuarterlyMT[, "interest_rate_us"])
  
  ## Fit model for the current sector
  Fit_QMT_3V <- auto.arima(ts_QuarterlyMT[, sector], xreg = Xreg_QMT_3Variables)
  
  ## Future values with LT data: fit trend models for each macroeconomic variable
  gdp_trend <- auto.arima(ts_QuarterlyLT[, "gdp_growth_rate"])     # GDP growth rate
  cpi_trend <- auto.arima(ts_QuarterlyLT[, "cpi"])                 # CPI
  ir_trend  <- auto.arima(ts_QuarterlyLT[, "interest_rate_us"])    # Interest Rate
  
  ### Forecast the next `h` quarters for macroeconomic variables
  future_gdp <- forecast(gdp_trend, h = h)$mean
  future_cpi <- forecast(cpi_trend, h = h)$mean
  future_ir  <- forecast(ir_trend, h = h)$mean
  
  ### Combine into future matrix for forecasting
  future_macro_trend_LT_3V <- cbind(future_gdp, future_cpi, future_ir)
  
  # Forecast the sector values
  future_forecast <- forecast(Fit_QMT_3V, xreg = future_macro_trend_LT_3V)
  
  # Prepare the forecast results and save as a data frame
  forecast_df <- data.frame(
    Date = seq.Date(from = as.Date("2025-01-01"), by = "quarter", length.out = h),
    Predicted_Sector = future_forecast$mean
  )
  
  # Create plot
  sector_plot <- autoplot(future_forecast) + ylab(paste(sector, "Forecast"))
  
  return(sector_plot)  # Return the plot for later use
}

# Loop through sectors and collect plots
sector_plots <- lapply(sectors, forecast_sector)

# Combine the plots into a single image
grid.arrange(grobs = sector_plots, ncol = 2)  # Adjust ncol (number of columns) as needed


# Save the combined plot as an image
ggsave("C:/Users/Paolo/Desktop/Project/combined_forecasts.png", 
       plot = grid.arrange(grobs = sector_plots, ncol = 2), 
       width = 12, height = 8)











# 2) MODEL WITH 3 VARIABLES: GDP GROWTH, CPI AND IRs
## Sectors
sectors <- c("avg_technology",
		 "avg_financials",
		 "avg_healthcare",
		 "avg_consumer_discretionary",
		 "avg_utilities",
		 "avg_industrials",
		 "avg_consumer_staples",
		 "avg_gold_us_price",
		 "avg_sandp_close",
		 "avg_bitcoin_close")

# Define function to forecast for each sector
forecast_sector <- function(sector) {

## Regressors, no harmonic
Xreg_QMT_3Variables <- cbind(GDPGrowth = ts_QuarterlyMT[, "gdp_growth_rate"], 
			     CPI = ts_QuarterlyMT[, "cpi"],
			     InterestRate = ts_QuarterlyMT[, "interest_rate_us"])
			

## Fit model, no harmonic
Fit_QMT_3V <- auto.arima(ts_QuarterlyMT[, sector], xreg = Xreg_QMT_3Variables)



## Future values with LT data: fit trend models for each macroeconomic variable
gdp_trend <- auto.arima(ts_QuarterlyLT[, "gdp_growth_rate"])     # GDP growth rate
cpi_trend <- auto.arima(ts_QuarterlyLT[, "cpi"])                 # CPI
ir_trend  <- auto.arima(ts_QuarterlyLT[, "interest_rate_us"])    # IR
gold_trend <- auto.arima(ts_QuarterlyMT[, "avg_gold_us_price"])  # Gold
vix_trend <- auto.arima(ts_QuarterlyMT[, "avg_vix_close"])       # VIX
eurusd_trend <- auto.arima(ts_QuarterlyMT[, "euro_dollar"])      # euro/dollar


### Forecast the next `h` quarters 
future_gdp <- forecast(gdp_trend, h = h)$mean
future_cpi <- forecast(cpi_trend, h = h)$mean
future_ir  <- forecast(ir_trend, h = h)$mean
future_gold <- forecast(gold_trend, h = h)$mean
future_vix <- forecast(vix_trend, h = h)$mean
future_er <- forecast(eurusd_trend, h = h)$mean


### Combine into future matrix
future_macro_trend_LT_3V <- cbind(future_gdp, future_cpi, future_ir)

# Forecast the sector values
  future_forecast <- forecast(Fit_QMT_3V, xreg = future_macro_trend_LT_3V)
  
  # Prepare the forecast results and save as a data frame
  forecast_df <- data.frame(
    Date = seq.Date(from = as.Date("2025-01-01"), by = "quarter", length.out = h),
    Predicted_Sector = future_forecast$mean
  )
  
  # Save the forecast to an Excel file
  file_name <- paste0("Forecast_", sector, "_ARIMA_Forecasts.xlsx")
  write_xlsx(forecast_df, paste0("C:/Users/Paolo/Desktop/Project/", file_name))
  
  # Plot the forecast
  autoplot(future_forecast) + ylab(paste(sector, "Forecast"))
  
  return(forecast_df)  # Return the forecast data for further use if needed
}

# Loop through sectors and forecast
all_forecasts <- lapply(sectors, forecast_sector)




# Combine the plots into a single image
grid.arrange(grobs = sector_plots, ncol = 2)  # Adjust ncol (number of columns) as needed




### Now forecast again
forecast(Fit_QMT_3V, xreg = future_macro_trend_LT_3V) %>% autoplot() + ylab("Technology stocks") # this gave better forecast so use future_macro_LT


summary(Fit_QMT_3V) # CONCLUSION: MODEL WITH 3 VARIABLES BETTER BECAUSE OF LOWER ERROR TERMS, LOWER AICc. NOW USE 3 VARIABLES MODEL TO TRY HARMONIC SERIES






# 3) MODEL WITH 3 VARIABLES AND HARMONIC REGRESSORS

## Set the value of K
K = 2

## Fourier terms for the training period
Fourier_Terms <- fourier(ts_QuarterlyMT[, "avg_technology"], K = K)

Xreg_QMT_3V_Fourier <- cbind(Xreg_QMT_3Variables, Fourier_Terms)


Fit_QMT_3V_Fourier <- auto.arima(ts_QuarterlyMT[, "avg_technology"], xreg = Xreg_QMT_3V_Fourier)


## Future terms
### Generate future fourier terms
Future_Fourier_Terms <- fourier(ts_QuarterlyMT[, "avg_technology"], K = K, h = h)

### Combine with macro variables
Future_Xreg_Fourier <- cbind(future_macro_trend_LT_3V, Future_Fourier_Terms)

summary(Fit_QMT_3V_Fourier) #AICc slightly better for K=2, but the one with no harmonic series is better


## Forecast with harmonic series
forecast(Fit_QMT_3V_Fourier, xreg = Future_Xreg_Fourier) %>% autoplot() + ylab("Technology stocks") # 



# 4) MODEL WITH ONLY HARMONIC SERIES

K <- 2

## Fourier terms for the training period
Fourier_Terms_Only <- fourier(ts_QuarterlyMT[, "avg_technology"], K = K)

## Fit the ARIMA model with only harmonic series
Fit_Fourier_Only <- auto.arima(ts_QuarterlyMT[, "avg_technology"], xreg = Fourier_Terms_Only)

## Future harmonic terms
Future_Fourier_Only <- fourier(ts_QuarterlyMT[, "avg_technology"], K = K, h = h)

## Forecast with only harmonic series
forecast(Fit_Fourier_Only, xreg = Future_Fourier_Only) %>% autoplot() + ylab("Technology stocks") # 

summary(Fit_Fourier_Only)




# REPEAT MODELS FOR THE MONTHLY DATA:
# Try the following models with monthly data:
# 1) GDP growth, CPI, IRs no harmonic series, 2) 3 Variables with harmonic series, 3) Only harmonic series

# 1) GDP growth, CPI, IRs no harmonic series

## Set the value of h (5 years * 12 months)
h_months <- 60

## Regressors, no harmonic
Xreg_MMT <- cbind(GDPGrowth = ts_MonthlyMT[, "gdp_growth_rate"], 
			CPI = ts_MonthlyMT[, "interpolated_cpi"],
			InterestRate = ts_MonthlyMT[, "interest_rate_us"])

## Fit model, no harmonic
Fit_MMT <- auto.arima(ts_MonthlyMT[, "avg_technology"], xreg = Xreg_MMT)



## Future values with LT data: fit trend models for each macroeconomic variable
gdp_trend_m <- auto.arima(ts_MonthlyLT[, "gdp_growth_rate"])     # GDP growth rate
cpi_trend_m <- auto.arima(ts_MonthlyLT[, "interpolated_cpi"])    # CPI
ir_trend_m  <- auto.arima(ts_MonthlyLT[, "interest_rate_us"])    # IR


### Forecast the next `h` months 
future_months_gdp <- forecast(gdp_trend_m, h = h_months)$mean
future_months_cpi <- forecast(cpi_trend_m, h = h_months)$mean
future_months_ir  <- forecast(ir_trend_m, h = h_months)$mean



### Combine into future matrix
future_macro_trend_LT_months <- cbind(future_months_gdp, future_months_cpi, future_months_ir)


### Forecast
forecast(Fit_MMT, xreg = future_macro_trend_LT_months) %>% autoplot() + ylab("Technology stocks") # this gave better forecast so use future_macro_LT


summary(Fit_MMT) # same model with quarterly data is better



# 2) 3 Variables with harmonic series

## Set the value of K
K_m <- 3

## Fourier terms for the training period
Fourier_Terms_m <- fourier(ts_MonthlyMT[, "avg_technology"], K = K_m)

Xreg_MMT_Fourier <- cbind(Xreg_MMT, Fourier_Terms_m)


Fit_MMT_Fourier <- auto.arima(ts_MonthlyMT[, "avg_technology"], xreg = Xreg_MMT_Fourier)


## Future terms
### Generate future fourier terms
Future_Fourier_Terms_m <- fourier(ts_MonthlyMT[, "avg_technology"], K = K_m, h = h_months)

### Combine with macro variables
Future_Xreg_Fourier_m <- cbind(future_macro_trend_LT_months, Future_Fourier_Terms_m)

summary(Fit_MMT_Fourier) #AICc slightly better for K=2, but the one with no harmonic series is better


## Forecast with harmonic series
forecast(Fit_MMT_Fourier, xreg = Future_Xreg_Fourier_m) %>% autoplot() + ylab("Technology stocks") # min AICc when K = 2, however still much higher than quarterly data models



# 3) Just harmonic series

K_m <- 2

## Fourier terms for the training period
Fourier_Terms_Only_m <- fourier(ts_MonthlyMT[, "avg_technology"], K = K_m)

## Fit the ARIMA model with only harmonic series
Fit_Fourier_Only_m <- auto.arima(ts_MonthlyMT[, "avg_technology"], xreg = Fourier_Terms_Only_m)

## Future harmonic terms
Future_Fourier_Only_m <- fourier(ts_MonthlyMT[, "avg_technology"], K = K_m, h = h_months)

## Forecast with only harmonic series
forecast(Fit_Fourier_Only_m, xreg = Future_Fourier_Only_m) %>% autoplot() + ylab("Technology stocks") # 

summary(Fit_Fourier_Only_m)




# NEXT STEP: COMPARE MODEL WITH 3 VARIABLES AND NO HARMONIC SERIES WITH RANDOM FOREST/MACHINE LEARNING MODELS
# Idea: try 2 ML models: 1) glmnet and 2) RF

# 1) set seed, 2) split into train and test, 3) x and y variables, 4) cross validation setup, 5) models

## 1) set seed
set.seed(123)


## 2) split into train and test
train_index <- sample(1:nrow(Data_Quarterly_MT), 0.8 * nrow(Data_Quarterly_MT))
train_data <- Data_Quarterly_MT[train_index, ]
test_data  <- Data_Quarterly_MT[-train_index, ]

train_data <- as.data.frame(train_data)
test_data  <- as.data.frame(test_data)

### Remove "quarter_date" from train_data and test_data
train_data <- train_data[, !colnames(train_data) %in% "quarter_date"]
test_data  <- test_data[, !colnames(test_data) %in% "quarter_date"]

## 3) x and y variables
X_train <- data.frame(
  GDPGrowth = train_data$gdp_growth_rate,
  CPI = train_data$cpi,
  InterestRate = train_data$interest_rate_us
)

Y_train_tech <- train_data$avg_technology


X_test <- data.frame(
  GDPGrowth = test_data$gdp_growth_rate,
  CPI = test_data$cpi,
  InterestRate = test_data$interest_rate_us
)

Y_test_tech <- test_data$avg_technology


## 4) cross validation setup
Train_Control <- trainControl(method = "cv", number = 5, savePredictions = "final")


## 5) Models
### a) glmnet model
glmnet_model_tech <- train(
  x = X_train,
  y = Y_train_tech,
  method = "glmnet",
  trControl = Train_Control,
  tuneLength = 10  # Try 10 different lambda values
)


print(glmnet_model_tech)


### b) Random Forest
rf_model_tech <- train(
  x = X_train,
  y = Y_train_tech,
  method = "ranger",
  trControl = Train_Control,
  tuneGrid = expand.grid(mtry = 1:3,  # Number of variables at each split
                         splitrule = "variance",
                         min.node.size = 5),  # Min node size
  num.trees = 500
)


print(rf_model_tech)


## 6) compare models
ML_models_list <- list(GLMNET = glmnet_model_tech, RF = rf_model_tech)

ML_models_results <- resamples(ML_models_list)

summary(ML_models_results) # Result: RF better due to lower error terms and higher R^2

### Box plots
bwplot(ML_models_results, metric = "Rsquared")
bwplot(ML_models_results, metric = "RMSE")
bwplot(ML_models_results, metric = "MAE")

### scatter plot
xyplot(ML_models_results)



## 7) check how they perform on test data

### Make predictions on the test set
pred_glmnet_tech <- predict(glmnet_model_tech, newdata = X_test)

pred_rf_tech <- predict(rf_model_tech, newdata = X_test)


### Calculate RMSE and MAE for both models
rmse_glmnet_tech <- sqrt(mean((pred_glmnet_tech - Y_test_tech)^2))
mae_glmnet_tech  <- mean(abs(pred_glmnet_tech - Y_test_tech))

rmse_rf_tech <- sqrt(mean((pred_rf_tech - Y_test_tech)^2))
mae_rf_tech  <- mean(abs(pred_rf_tech - Y_test_tech))

### Print results
cat("\nGLMNET RMSE:", rmse_glmnet_tech, "MAE:", mae_glmnet_tech, "\n")
cat("Random Forest RMSE:", rmse_rf_tech, "MAE:", mae_rf_tech, "\n") # Result: RF model better on the test set


## 8) Fine tuning the RF model

### Set up a grid of hyperparameters
tuneGrid <- expand.grid(
  .mtry = c(2, 3, 4, 5),              # number of variables to try at each split
  .ntree = c(500, 1000),              # number of trees in the forest
  .nodesize = c(1, 5, 10)             # minimum node size
)

tuneGrid <- expand.grid(
  .mtry = c(2, 3, 4, 5)  # number of variables to try at each split
)

### Fit the Random Forest model with cross-validation
rf_tuned_tech <- train(
  avg_technology ~ gdp_growth_rate + cpi + interest_rate_us,  # Formula for the model
  data = train_data,
  method = "rf",                  # Use Random Forest method
  trControl = Train_Control,      # Cross-validation setup
  tuneGrid = tuneGrid,            # Hyperparameter grid
  ntree = 500,			    # Number of trees in forest
  nodesize = 5,			    # Minimum node size
  importance = TRUE               # Calculate variable importance
)

print(rf_tuned_tech)

varImpPlot(rf_tuned_tech$finalModel)


best_params_tech <- rf_tuned_tech$bestTune
print(best_params_tech)
plot(rf_tuned_tech)
importance(rf_tuned_tech$finalModel)
summary(rf_tuned_tech)

## 9) Use the tuned model to predict on the test set
rf_pred_tech <- predict(rf_tuned_tech, newdata = test_data)

### Calculate RMSE and MAE for the predictions
rmse_rf_tuned_tech <- sqrt(mean((rf_pred_tech - test_data$avg_technology)^2))
mae_rf_tuned_tech <- mean(abs(rf_pred_tech - test_data$avg_technology))

cat("Tuned Random Forest RMSE:", rmse_rf_tuned_tech, "MAE:", mae_rf_tuned_tech, "\n") 





## 10) Retry with more, different variables
# Rename columns in train_data
colnames(train_data)[colnames(train_data) == "rgdp2017"] <- "GDP"
colnames(train_data)[colnames(train_data) == "avg_vix_close"] <- "VIX"

### Adding more variables to the model formula
rf_tuned_tech_more_vars <- train(
  avg_technology ~ GDP + VIX,  # best model seems to be with gold and unemployment rate
  data = train_data,
  method = "rf",                  # Use Random Forest method
  trControl = Train_Control,      # Cross-validation setup
  tuneGrid = tuneGrid,            # Hyperparameter grid
  ntree = 500,                    # Number of trees in the forest
  nodesize = 5,                   # Minimum node size
  importance = TRUE               # Calculate variable importance
)

### Print out the best model and its tuning parameters
print(rf_tuned_tech_more_vars)

### Evaluate the performance of the model on test data
predictions_rf_tuned_more_vars <- predict(rf_tuned_tech_more_vars, newdata = test_data)
rmse_rf_tuned_more_vars <- sqrt(mean((predictions_rf_tuned_more_vars - test_data$avg_technology)^2))
mae_rf_tuned_more_vars <- mean(abs(predictions_rf_tuned_more_vars - test_data$avg_technology))

# Print the performance metrics
cat("Tuned Random Forest with more variables RMSE:", rmse_rf_tuned_more_vars, "MAE:", mae_rf_tuned_more_vars, "\n")


varImpPlot(rf_tuned_tech_more_vars$finalModel)


# Extract variable importance using caret
rf_importance <- varImp(rf_tuned_tech_more_vars, scale = TRUE)
print(rf_importance)

# Plot the variable importance
plot(rf_importance)




# SITUATION: GDP GROWTH, CPI, IR: GLMNet model better. Choosing different variables in RF makes model better
# Compare models with the different variables?
- GLMNet GDP, VIX
- RF GDP, VIX




X_train_NewVariables <- data.frame(
  GDP = train_data$rgdp2017,
  VIX = train_data$avg_vix_close)


glmnet_model_tech_NewVariables <- train(
  x = X_train_NewVariables,
  y = Y_train_tech,
  method = "glmnet",
  trControl = Train_Control,
  tuneLength = 10  # Try 10 different lambda values
)

ML_models_NewVariables <- list(GLMNET = glmnet_model_tech_NewVariables, RF = rf_tuned_tech_more_vars)

ML_models_NewVariablesResults <- resamples(ML_models_NewVariables)

summary(ML_models_NewVariablesResults) # Result: RF better due to lower error terms and higher R^2

### Box plots
bwplot(ML_models_NewVariablesResults, metric = "Rsquared")
bwplot(ML_models_NewVariablesResults, metric = "RMSE")
bwplot(ML_models_NewVariablesResults, metric = "MAE")

### scatter plot
xyplot(ML_models_NewVariablesResults)


## Check models on the test set to check for overfitting
# Make predictions on the test set
# Convert the relevant variables to matrix format
test_matrix_glmnet <- as.matrix(test_data[, c("rgdp2017", "avg_vix_close")])

# Make predictions
glmnet_test_predictions_NV <- predict(glmnet_model_tech_NewVariables, 
                                      newdata = test_data[, c("rgdp2017", "avg_vix_close")])

rf_test_predictions_NV <- predict(rf_tuned_tech_more_vars, newdata = test_data)


# Calculate RMSE and MAE for both models on test data
rf_rmse_test_NV <- sqrt(mean((rf_test_predictions_NV - test_data$avg_technology)^2))
rf_mae_test_NV <- mean(abs(rf_test_predictions_NV - test_data$avg_technology))

glmnet_rmse_test_NV <- sqrt(mean((glmnet_test_predictions_NV - test_data$avg_technology)^2))
glmnet_mae_test_NV <- mean(abs(glmnet_test_predictions_NV - test_data$avg_technology))

# Print results
cat("\nRF Test Set RMSE:", rf_rmse_test_NV, "MAE:", rf_mae_test_NV, "\n")
cat("GLMNET Test Set RMSE:", glmnet_rmse_test_NV, "MAE:", glmnet_mae_test_NV, "\n")



# Extract the final glmnet model
glmnet_final_model <- glmnet_model_tech_NewVariables$finalModel

# Retrieve the coefficients at the best lambda value
coef(glmnet_final_model, s = glmnet_model_tech_NewVariables$bestTune$lambda)

glmnet_test_predictions_NV <- predict(glmnet_model_tech_NewVariables, 
                                      newdata = test_data[, c("rgdp2017", "avg_vix_close")])

# Rename columns in test_data to match model variable names
colnames(test_data)[colnames(test_data) == "rgdp2017"] <- "GDP"
colnames(test_data)[colnames(test_data) == "avg_vix_close"] <- "VIX"

glmnet_test_predictions_NV <- predict(glmnet_model_tech_NewVariables, 
                                      newdata = test_data[, c("GDP", "VIX")])

nrow(test_data)                     # Should be 21
length(glmnet_test_predictions_NV)  # Should now also be 21



# Visualise comparison
# Create a comparison dataframe
comparison_df_NV <- data.frame(
  Date = test_data$date,
  Actual = test_data$avg_technology,
  RF_Predicted = rf_test_predictions_NV,
  GLMNET_Predicted = glmnet_test_predictions_NV
)

# Plot the predictions vs. actuals
ggplot(comparison_df_NV, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1.2) +
  geom_line(aes(y = RF_Predicted, color = "RF Predicted"), size = 1.2) +
  geom_line(aes(y = GLMNET_Predicted, color = "GLMNET Predicted"), size = 1.2) +
  labs(title = "Model Comparison: RF vs GLMNET on Test Data",
       x = "Date", y = "Technology Stocks") +
  scale_color_manual(values = c("Actual" = "black", "RF Predicted" = "blue", "GLMNET Predicted" = "red")) +
  theme_minimal()


# Check the dimensions
nrow(test_data)                    # Rows in the actual test set
length(rf_test_predictions_NV)     # Rows in RF predictions
length(glmnet_test_predictions_NV) # Rows in GLMNET predictions



#### RESULT: RF MODEL VS GLMNET WITH GDP AND VIX: RF BETTER DUE TO LOWER ERROR TERMS AND FITS BETTER, CHECKED ON TEST DATA FOR OVERFITTING AND ERRORS REMAIN LOW






# COMPARE ARIMA MODEL WITH NO HARMONIC SERIES WITH RF MODEL
## 1) Forecast using ARMIMA model, 2) predict() with RF, 3) Compare, 4) Visualise comparison

## 1) Forecast using ARMIMA model
### Forecast using the ARIMA model with economic variables
arima_forecast <- forecast(Fit_QMT_3V, xreg = future_macro_trend_LT_3V, h = nrow(test_data))

### Extract the mean predictions
predictions_arima <- as.numeric(arima_forecast$mean)

### Calculate RMSE and MAE for the ARIMA model
rmse_arima <- sqrt(mean((predictions_arima - test_data$avg_technology)^2))
mae_arima <- mean(abs(predictions_arima - test_data$avg_technology))

### Print the performance metrics
cat("ARIMA RMSE:", rmse_arima, "MAE:", mae_arima, "\n")


## 2) predict() with RF
rf_pred_tech <- predict(rf_tuned_tech, newdata = test_data)

### Calculate RMSE and MAE for the predictions
rmse_rf_tuned_tech <- sqrt(mean((rf_pred_tech - test_data$avg_technology)^2))
mae_rf_tuned_tech <- mean(abs(rf_pred_tech - test_data$avg_technology))

cat("Tuned Random Forest RMSE:", rmse_rf_tuned_tech, "MAE:", mae_rf_tuned_tech, "\n") # Lower values than RF before



## 3) Compare

### Create a data frame with actual values and predictions
comparison_df <- data.frame(
  Date = test_data$date,
  Actual = test_data$avg_technology,
  RF_Predicted = rf_pred_tech,
  ARIMA_Predicted = predictions_arima
)


ggplot(comparison_df, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1.2) +
  geom_line(aes(y = RF_Predicted, color = "Random Forest"), size = 1.2) +
  geom_line(aes(y = ARIMA_Predicted, color = "ARIMA"), size = 1.2) +
  labs(title = "Model Comparison: Random Forest vs ARIMA",
       x = "Date", y = "Technology Stocks") +
  scale_color_manual(values = c("Actual" = "black", "Random Forest" = "blue", "ARIMA" = "red")) +
  theme_minimal()

length(test_dates)           # Number of test dates
nrow(test_data)              # Number of rows in test_data
length(rf_pred_tech)        # Number of Random Forest predictions
length(predictions_arima)   # Number of ARIMA predictions




## 4) Plot RF predictions
### dataframe with actual values and RF predictions
rf_comparison_df <- data.frame(
  Date = test_data$date,
  Actual = test_data$avg_technology,
  RF_Predicted = rf_pred_tech
)

### plot
ggplot(rf_comparison_df, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = RF_Predicted, color = "Random Forest Prediction"), size = 1, linetype = "dashed") +
  labs(title = "Actual vs Predicted: Random Forest",
       x = "Date", y = "Technology Stock Price") +
  scale_color_manual(values = c("Actual" = "black", "Random Forest Prediction" = "blue")) +
  theme_minimal()


# Calculate residuals
rf_comparison_df$residuals <- rf_comparison_df$Actual - rf_comparison_df$RF_Predicted

# Plot residuals
ggplot(rf_comparison_df, aes(x = Date, y = residuals)) +
  geom_line(color = "red") +
  labs(title = "Residuals: Random Forest Model",
       x = "Date", y = "Residuals (Actual - Predicted)") +
  theme_minimal()






# ARIMA MODEL VS RF MODEL WITH GDP AND VIX
## ARIMA with GDP and VIX
# 2) MODEL WITH 2 VARIABLES: GDP, VIX
## Regressors, no harmonic
Xreg_QMT_2Variables <- cbind(GDP = ts_QuarterlyMT[, "rgdp2017"], 
			     VIX = ts_QuarterlyMT[, "avg_vix_close"])
			

## Fit model, no harmonic
Fit_QMT_2V <- auto.arima(ts_QuarterlyMT[, "avg_technology"], xreg = Xreg_QMT_2Variables)



## Future values with LT data: fit trend models for each macroeconomic variable
rgdp_trend <- auto.arima(ts_QuarterlyLT[, "rgdp2017"])	     # Real GDP
gdp_trend <- auto.arima(ts_QuarterlyLT[, "gdp_growth_rate"])     # GDP growth rate
cpi_trend <- auto.arima(ts_QuarterlyLT[, "cpi"])                 # CPI
ir_trend  <- auto.arima(ts_QuarterlyLT[, "interest_rate_us"])    # IR
gold_trend <- auto.arima(ts_QuarterlyMT[, "avg_gold_us_price"])  # Gold
vix_trend <- auto.arima(ts_QuarterlyMT[, "avg_vix_close"])       # VIX
eurusd_trend <- auto.arima(ts_QuarterlyMT[, "euro_dollar"])      # euro/dollar


### Forecast the next `h` quarters 
future_rgdp <- forecast(rgdp_trend, h = h)$mean
future_gdp <- forecast(gdp_trend, h = h)$mean
future_cpi <- forecast(cpi_trend, h = h)$mean
future_ir  <- forecast(ir_trend, h = h)$mean
future_gold <- forecast(gold_trend, h = h)$mean
future_vix <- forecast(vix_trend, h = h)$mean
future_er <- forecast(eurusd_trend, h = h)$mean


### Combine into future matrix
future_macro_trend_LT_2V <- cbind(future_rgdp, future_vix)


### Now forecast again
ARIMA_forecast_2V <- forecast(Fit_QMT_2V, xreg = future_macro_trend_LT_2V)

forecast(Fit_QMT_2V, xreg = future_macro_trend_LT_2V) %>% autoplot() + ylab("Technology stocks") # this gave better forecast so use future_macro_LT


summary(Fit_QMT_2V) 



# Same model with harmonic series
## Set the value of K
K = 2

## Fourier terms for the training period
Fourier_Terms <- fourier(ts_QuarterlyMT[, "avg_technology"], K = K)

Xreg_QMT_2V_Fourier <- cbind(Xreg_QMT_2Variables, Fourier_Terms)


Fit_QMT_2V_Fourier <- auto.arima(ts_QuarterlyMT[, "avg_technology"], xreg = Xreg_QMT_2V_Fourier)


## Future terms
### Generate future fourier terms
Future_Fourier_Terms <- fourier(ts_QuarterlyMT[, "avg_technology"], K = K, h = h)

### Combine with macro variables
Future_Xreg_Fourier_2V <- cbind(future_macro_trend_LT_2V, Future_Fourier_Terms)

summary(Fit_QMT_2V_Fourier) #AICc slightly better for K=2, but the one with no harmonic series is better


## Forecast with harmonic series
ARIMA_forecast_2V_Fourier <- forecast(Fit_QMT_2V_Fourier, xreg = Future_Xreg_Fourier_2V)

forecast(Fit_QMT_2V_Fourier, xreg = Future_Xreg_Fourier_2V) %>% autoplot() + ylab("Technology stocks") # 


length(ARIMA_forecast_2V_Fourier_predictions)  # Check the length of ARIMA predictions
nrow(test_data)  # Check the number of rows in test_data




# COMPARE RF AND ARIMA K=2, GDP AND VIX
## Predictions
ARIMA_forecast_2V_Fourier_predictions <- as.numeric(ARIMA_forecast_2V_Fourier$mean)
predictions_rf_tuned_more_vars <- predict(rf_tuned_tech_more_vars, newdata = test_data)


## Errors
### Calculate RMSE and MAE for both models on test data
rf_rmse_test_NV <- sqrt(mean((rf_test_predictions_NV - test_data$avg_technology)^2))
rf_mae_test_NV <- mean(abs(rf_test_predictions_NV - test_data$avg_technology))

### ARIMA Errors
arima_rmse_test_2VF <- sqrt(mean((ARIMA_forecast_2V_Fourier_predictions - test_data$avg_technology)^2))
arima_mae_test_2VF <- mean(abs(ARIMA_forecast_2V_Fourier_predictions - test_data$avg_technology))


## Print results
cat("\nRF Test Set RMSE:", rf_rmse_test_NV, "MAE:", rf_mae_test_NV, "\n")
cat("\nARIMA Test Set RMSE:", arima_rmse_test_2VF, "MAE:", arima_mae_test_2VF)


### Create a data frame with actual values and predictions
comparison_df_2V <- data.frame(
  Date = test_data$date,
  Actual = test_data$avg_technology,
  RF_Predicted = predictions_rf_tuned_more_vars,
  ARIMA_Predicted = ARIMA_forecast_2V_Fourier_predictions
)


ggplot(comparison_df_2V, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1.2) +
  geom_line(aes(y = rf_test_predictions_NV, color = "Random Forest"), size = 1.2) +
  geom_line(aes(y = ARIMA_forecast_2V_Fourier_predictions, color = "ARIMA"), size = 1.2) +
  labs(title = "Model Comparison: Random Forest vs ARIMA",
       x = "Date", y = "Technology Stocks") +
  scale_color_manual(values = c("Actual" = "black", "Random Forest" = "blue", "ARIMA" = "red")) +
  theme_minimal()

length(test_dates)           # Number of test dates
nrow(test_data)              # Number of rows in test_data
length(rf_pred_tech)        # Number of Random Forest predictions
length(predictions_arima)   # Number of ARIMA predictions






##### RESULT: ARIMA MODEL HAS MUCH LARGER ERRORS ON TEST DATA WHICH SUGGESTS OVERFITTING, RF MODEL BETTER

# Q: WHAT IF I COMPARE RF MODEL WITH ARIMA WITH NO HARMONIC SERIES? DOES THAT HELP WITH OVERFITTING?

# COMPARE RF AND ARIMA no harmonic, GDP AND VIX
## Predictions
ARIMA_forecast_2V <- as.numeric(ARIMA_forecast_2V$mean)
predictions_rf_tuned_more_vars <- predict(rf_tuned_tech_more_vars, newdata = test_data)


## Errors
### Calculate RMSE and MAE for both models on test data
rf_rmse_test_NV <- sqrt(mean((rf_test_predictions_NV - test_data$avg_technology)^2))
rf_mae_test_NV <- mean(abs(rf_test_predictions_NV - test_data$avg_technology))

### ARIMA Errors
arima_rmse_test_2V <- sqrt(mean((ARIMA_forecast_2V - test_data$avg_technology)^2))
arima_mae_test_2V <- mean(abs(ARIMA_forecast_2V - test_data$avg_technology)) # slightly lower errors compared to harmonic series





### RE-DO ARIMA FORECAST WITH THE MEDIUM TERM DATA, COULD GIVE LOWER ERRORS ON THE TEST DATA

## Future values with MT data: fit trend models for each macroeconomic variable
rgdp_trend_MT <- auto.arima(ts_QuarterlyMT[, "rgdp2017"])	     # Real GDP
vix_trend_MT <- auto.arima(ts_QuarterlyMT[, "avg_vix_close"])    # VIX

### Forecast the next `h` quarters 
future_rgdp_MT <- forecast(rgdp_trend_MT, h = h)$mean
future_vix_MT <- forecast(vix_trend_MT, h = h)$mean


### Combine into future matrix
future_macro_trend_MT_2V <- cbind(future_rgdp_MT, future_vix_MT)


### Now forecast again
ARIMA_forecast_2V_MT <- forecast(Fit_QMT_2V, xreg = future_macro_trend_MT_2V)

forecast(Fit_QMT_2V, xreg = future_macro_trend_MT_2V) %>% autoplot() + ylab("Technology stocks") # this gave better forecast so use future_macro_LT


summary(Fit_QMT_2V)


# Compare MT ARIMA to RF
## Predictions
ARIMA_forecast_2V_MT <- as.numeric(ARIMA_forecast_2V_MT$mean)
predictions_rf_tuned_more_vars <- predict(rf_tuned_tech_more_vars, newdata = test_data)


## Errors
### Calculate RMSE and MAE for both models on test data
rf_rmse_test_NV <- sqrt(mean((rf_test_predictions_NV - test_data$avg_technology)^2))
rf_mae_test_NV <- mean(abs(rf_test_predictions_NV - test_data$avg_technology))

### ARIMA Errors
arima_rmse_test_2V_MT <- sqrt(mean((ARIMA_forecast_2V_MT - test_data$avg_technology)^2))
arima_mae_test_2V_MT <- mean(abs(ARIMA_forecast_2V_MT - test_data$avg_technology)) # slightly lower errors compared to harmonic series

## Print results
cat("\nRF Test Set RMSE:", rf_rmse_test_NV, "MAE:", rf_mae_test_NV, "\n")
cat("\nARIMA Test Set RMSE:", arima_rmse_test_2VF_MT, "MAE:", arima_mae_test_2VF_MT)


# Adjust the length of rf_test_predictions_NV to match the length of the actual values
rf_test_predictions_NV <- rf_test_predictions_NV[1:length(test_data$avg_technology)]

# Now calculate the MAE and RMSE
rf_mae_test_NV <- mean(abs(rf_test_predictions_NV - test_data$avg_technology))
rf_rmse_test_NV <- sqrt(mean((rf_test_predictions_NV - test_data$avg_technology)^2))

cat("\nRF Test Set RMSE:", rf_rmse_test_NV, "MAE:", rf_mae_test_NV, "\n")








## 4) Plot RF predictions
### dataframe with actual values and RF predictions
rf_comparison_df <- data.frame(
  Date = test_data$date,
  Actual = test_data$avg_technology,
  RF_Predicted = rf_pred_tech
)

### plot
ggplot(rf_comparison_df, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = RF_Predicted, color = "Random Forest Prediction"), size = 1, linetype = "dashed") +
  labs(title = "Actual vs Predicted: Random Forest",
       x = "Date", y = "Technology Stock Price") +
  scale_color_manual(values = c("Actual" = "black", "Random Forest Prediction" = "blue")) +
  theme_minimal()


# Calculate residuals
rf_comparison_df$residuals <- rf_comparison_df$Actual - rf_comparison_df$RF_Predicted

# Plot residuals
ggplot(rf_comparison_df, aes(x = Date, y = residuals)) +
  geom_line(color = "red") +
  labs(title = "Residuals: Random Forest Model",
       x = "Date", y = "Residuals (Actual - Predicted)") +
  theme_minimal()
