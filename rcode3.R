## 1) set seed
set.seed(123)

h <- 20

### Forecast the next `h` quarters 
future_rgdp_MT <- forecast(rgdp_trend_MT, h = h)$mean
future_vix_MT <- forecast(vix_trend_MT, h = h)$mean


future_rgdp <- forecast(rgdp_trend_MT, h = h)$mean
future_vix <- forecast(vix_trend_MT, h = h)$mean


# Combine into future macro data frame
future_macro_data <- data.frame(
  rgdp2017 = future_rgdp,
  avg_vix_close = future_vix
)


# Use the RF model to predict tech stocks for the next 5 years
future_rf_predictions <- predict(rf_tuned_tech_NV, newdata = future_macro_data)

# Display future predictions
future_rf_predictions


# Get the last actual price from the test data
last_actual_price <- tail(test_data$avg_technology, 1)

# Determine direction: 1 = Increase, 0 = Decrease
future_direction <- ifelse(future_rf_predictions > last_actual_price, 1, 0)

# Combine the results into a dataframe
future_results <- data.frame(
  Date = seq.Date(from = as.Date("2025-01-01"), by = "quarter", length.out = h),
  Predicted_Price = future_rf_predictions,
  Direction = ifelse(future_direction == 1, "Increase", "Decrease"),
  Magnitude = future_rf_predictions - last_actual_price
)

# Display the future results
print(future_results)





## TRY WITH DIFFERENT SECTORS
# List of sector columns
sectors <- c("avg_technology", "avg_financials", "avg_healthcare", 
             "avg_consumer_discretionary", "avg_utilities", 
             "avg_industrials", "avg_consumer_staples")

# Create a list to store the models and future predictions
rf_models <- list()
future_predictions <- list()

# Loop through each sector
for (sector in sectors) {
  # Train a separate RF model for each sector
  rf_model <- train(
    as.formula(paste(sector, "~ rgdp2017 + avg_vix_close")),  
    data = train_data,
    method = "rf",
    trControl = Train_Control,
    tuneGrid = tuneGrid,
    ntree = 500,
    nodesize = 5,
    importance = TRUE
  )
  
  # Store the model
  rf_models[[sector]] <- rf_model
  
  # Make future predictions using the sector-specific model
  future_predictions[[sector]] <- predict(rf_model, newdata = future_macro_data)
}

# Display future predictions for all sectors
future_predictions


# Create a dataframe with the results
future_sector_df <- data.frame(
  Date = seq.Date(from = as.Date("2025-01-01"), by = "quarter", length.out = h)
)

# Add predictions for each sector
for (sector in sectors) {
  future_sector_df[[sector]] <- future_predictions[[sector]]
}

# Display the future sector predictions
print(future_sector_df)


write_xlsx(future_sector_df, "C:/Users/Paolo/Desktop/Project/Future_RF_Predictions_Sectors.xlsx")



## RETRY MODEL WITHOUT VIX

# Add lag columns to both train and test data
train_data$avg_technology_lag1 <- dplyr::lag(train_data$avg_technology, 1)
train_data$avg_technology_lag2 <- dplyr::lag(train_data$avg_technology, 2)

test_data$avg_technology_lag1 <- dplyr::lag(test_data$avg_technology, 1)
test_data$avg_technology_lag2 <- dplyr::lag(test_data$avg_technology, 2)

# Remove rows with NAs due to lag
train_data_no_na <- na.omit(train_data)
test_data_no_na <- na.omit(test_data)

# Retrain the Random Forest model without VIX
rf_tuned_no_vix <- train(
  avg_technology ~ rgdp2017 + avg_technology_lag1 + avg_technology_lag2,  
  data = train_data_no_na,
  method = "rf",
  trControl = Train_Control,
  tuneGrid = tuneGrid,
  ntree = 500,
  nodesize = 5,
  mtry = 2,  # Set mtry explicitly to 2 (number of predictors)
  importance = TRUE
)

# Make predictions for training and test sets
rf_predictions_no_vix_train <- predict(rf_tuned_no_vix, newdata = train_data_no_na)
rf_predictions_no_vix_test <- predict(rf_tuned_no_vix, newdata = test_data_no_na)

# Calculate error metrics
rmse_no_vix_train <- sqrt(mean((rf_predictions_no_vix_train - train_data_no_na$avg_technology)^2))
mae_no_vix_train <- mean(abs(rf_predictions_no_vix_train - train_data_no_na$avg_technology))

rmse_no_vix_test <- sqrt(mean((rf_predictions_no_vix_test - test_data_no_na$avg_technology)^2))
mae_no_vix_test <- mean(abs(rf_predictions_no_vix_test - test_data_no_na$avg_technology))

# Print results
cat("\nTraining Set RMSE:", rmse_no_vix_train, "MAE:", mae_no_vix_train)
cat("\nTest Set RMSE:", rmse_no_vix_test, "MAE:", mae_no_vix_test)

# Forecasting future quarters without VIX
n_future <- 20  # 5 years (20 quarters)
future_forecasts_no_vix <- numeric(n_future)

# Initialize with the last known stock price and lags
last_known_price <- tail(test_data$avg_technology, 1)
last_lag1 <- tail(test_data$avg_technology_lag1, 1)
last_lag2 <- tail(test_data$avg_technology_lag2, 1)

# Future macroeconomic data (GDP only)
future_gdp <- future_macro_trend_MT_2V[, 1]  # Ensure future macro data is available

# Create future data frame
future_df_no_vix <- data.frame(
  rgdp2017 = future_gdp,
  avg_technology_lag1 = rep(NA, n_future),
  avg_technology_lag2 = rep(NA, n_future)
)

# Rolling forecast without VIX
for (i in 1:n_future) {
  # Update lags
  if (i == 1) {
    future_df_no_vix$avg_technology_lag1[i] <- last_known_price
    future_df_no_vix$avg_technology_lag2[i] <- last_lag1
  } else {
    future_df_no_vix$avg_technology_lag1[i] <- future_forecasts_no_vix[i - 1]
    future_df_no_vix$avg_technology_lag2[i] <- future_forecasts_no_vix[i - 2]
  }
  
  # Make predictions
  future_forecasts_no_vix[i] <- predict(rf_tuned_no_vix, newdata = future_df_no_vix[i, ])
}

# Create dataframe with predictions
future_forecast_df_no_vix <- data.frame(
  Date = seq.Date(from = as.Date("2025-01-01"), by = "quarter", length.out = n_future),
  Predicted_tech_stocks_no_vix = future_forecasts_no_vix
)

# Save to Excel
write_xlsx(future_forecast_df_no_vix, "C:/Users/Paolo/Desktop/Project/Future_RF_No_VIX_Predictions.xlsx")


rf_predictions_train <- predict(rf_tuned_no_vix, newdata = train_data_no_na)
plot(train_data_no_na$avg_technology, type = "l", col = "blue")
lines(rf_predictions_train, col = "red")
