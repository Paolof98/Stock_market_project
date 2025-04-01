# 1) ARIMA NO HARMONIC MEDIUM TERM

ts_QuarterlyMT <- ts(Data_Quarterly_MT[, 2:19], start = c(1999, 1), frequency = 4)

## Future values with MT data: fit trend models for each macroeconomic variable (GDP and VIX)
rgdp_trend_MT <- auto.arima(ts_QuarterlyMT[, "rgdp2017"])	     # Real GDP
vix_trend_MT <- auto.arima(ts_QuarterlyMT[, "avg_vix_close"])    # VIX

## forecast 5 years * 4 quarters = h = 20
h <- 20

### Forecast the next `h` quarters 
future_rgdp_MT <- forecast(rgdp_trend_MT, h = h)$mean
future_vix_MT <- forecast(vix_trend_MT, h = h)$mean

## Regressors, no harmonic
Xreg_QMT_2Variables <- cbind(rgdp2017 = ts_QuarterlyMT[, "rgdp2017"], 
			     avg_vix_close = ts_QuarterlyMT[, "avg_vix_close"])


### Combine into future matrix
future_macro_trend_MT_2V <- cbind(future_rgdp_MT, future_vix_MT)


## Fit model, no harmonic
Fit_QMT_2V <- auto.arima(ts_QuarterlyMT[, "avg_technology"], xreg = Xreg_QMT_2Variables)


### Now forecast
ARIMA_forecast_2V_MT <- forecast(Fit_QMT_2V, xreg = future_macro_trend_MT_2V)

forecast(Fit_QMT_2V, xreg = future_macro_trend_MT_2V) %>% autoplot() + ylab("Technology stocks") # this gave better forecast so use future_macro_LT


summary(Fit_QMT_2V)



# 2) RF WITH GDP AND VIX
# 1) set seed, 2) split into train and test, 3) x and y variables, 4) cross validation setup, 5) models

## 1) set seed
set.seed(123)


## 2) split into train and test
train_index <- sample(1:nrow(Data_Quarterly_MT), 0.8 * nrow(Data_Quarterly_MT))
train_data <- Data_Quarterltrain_index
y_MT[train_index, ]
test_data  <- Data_Quarterly_MT[-train_index, ]

train_data <- as.data.frame(train_data)
test_data  <- as.data.frame(test_data)

### Remove "quarter_date" from train_data and test_data
train_data <- train_data[, !colnames(train_data) %in% "quarter_date"]
test_data  <- test_data[, !colnames(test_data) %in% "quarter_date"]

## 3) x and y variables
X_train_NewVariables <- data.frame(
  rgdp2017 = train_data$rgdp2017,
  avg_vix_close = train_data$avg_vix_close)


Y_train_tech <- train_data$avg_technology


X_test_NewVariables <- data.frame(
  rgdp2017 = train_data$rgdp2017,
  avg_vix_close = train_data$avg_vix_close
)

Y_test_tech <- test_data$avg_technology

## 4) cross validation setup
Train_Control <- trainControl(method = "cv", number = 5, savePredictions = "final")

### Set up a grid of hyperparameters
tuneGrid <- expand.grid(
  .mtry = c(2, 3, 4, 5)  # number of variables to try at each split
)


### Adding more variables to the model formula
rf_tuned_tech_NV <- train(
  avg_technology ~ rgdp2017 + avg_vix_close,  # best model seems to be with gold and unemployment rate
  data = train_data,
  method = "rf",                  # Use Random Forest method
  trControl = Train_Control,      # Cross-validation setup
  tuneGrid = tuneGrid,            # Hyperparameter grid
  ntree = 500,                    # Number of trees in the forest
  nodesize = 5,                   # Minimum node size
  importance = TRUE               # Calculate variable importance
)

### Print out the best model and its tuning parameters
print(rf_tuned_tech_NV)



### Evaluate the performance of the model on test data
predictions_rf_tuned_NV <- predict(rf_tuned_tech_NV, newdata = test_data)
rmse_rf_tuned_NV <- sqrt(mean((predictions_rf_tuned_NV - test_data$avg_technology)^2))
mae_rf_tuned_NV <- mean(abs(predictions_rf_tuned_NV - test_data$avg_technology))

# Print the performance metrics
cat("Tuned Random Forest with more variables RMSE:", rmse_rf_tuned_NV, "MAE:", mae_rf_tuned_NV, "\n")




# Create a comparison dataframe
comparison_rf_test_df <- data.frame(
  Date = test_data$date,
  Actual = test_data$avg_technology,
  RF_Predicted = predictions_rf_tuned_NV
)

# Plot the actual vs predicted values
ggplot(comparison_rf_test_df, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1.2) +
  geom_line(aes(y = RF_Predicted, color = "RF Predicted"), size = 1.2) +
  labs(title = "Random Forest Model Predictions on Test Data",
       x = "Date", y = "Technology Stocks") +
  scale_color_manual(values = c("Actual" = "black", "RF Predicted" = "blue")) +
  theme_minimal()


# Extract training RMSE and MAE
rf_train_rmse <- min(rf_tuned_tech_NV$results$RMSE)
rf_train_mae <- min(rf_tuned_tech_NV$results$MAE)

# Print results
cat("\nTraining Set RMSE:", rf_train_rmse, "MAE:", rf_train_mae)

rmse_rf_tuned_NV <- sqrt(mean((predictions_rf_tuned_NV - test_data$avg_technology)^2))
mae_rf_tuned_NV <- mean(abs(predictions_rf_tuned_NV - test_data$avg_technology))

# Print the performance metrics
cat("Tuned Random Forest with more variables RMSE:", rmse_rf_tuned_NV, "MAE:", mae_rf_tuned_NV, "\n")









## COMPARE TRAIN AND TEST
cat("\nTraining Set RMSE:", rf_train_rmse, "MAE:", rf_train_mae)
cat("\nTest Set RMSE:", rmse_rf_tuned_NV, "MAE:", mae_rf_tuned_NV)


varImpPlot(rf_tuned_tech_NV$finalModel)


# Extract variable importance using caret
rf_importance <- varImp(rf_tuned_tech_more_vars, scale = TRUE)
print(rf_importance)


# Calculate residuals
residuals_rf <- test_data$avg_technology - predictions_rf_tuned_NV

# Plot residuals
ggplot(data = data.frame(Date = test_data$date, Residuals = residuals_rf), aes(x = Date, y = Residuals)) +
  geom_line(color = "red") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals Over Time (Test Set)", x = "Date", y = "Residuals") +
  theme_minimal()

rf_tuned_tech_NV$results

### RESULTS: NO PARTICULAR DIFFERENCE BETWEEN TRAIN AND TEST ERRORS (IN FACT TEST ERRORS ARE SMALLER), THIS SUGGESTS THAT MODEL IS NOT OVERFITTING










# 3) COMPARE BOTH MODELS ON PREDICTIONS
## Predictions
ARIMA_forecast_2V_MT <- as.numeric(ARIMA_forecast_2V_MT$mean)
predictions_rf_tuned_NV <- predict(rf_tuned_tech_NV, newdata = test_data)


## Errors
### Calculate RMSE and MAE for both models on test data
rmse_rf_tuned_NV <- sqrt(mean((predictions_rf_tuned_NV - test_data$avg_technology)^2))
mae_rf_tuned_NV <- mean(abs(predictions_rf_tuned_NV - test_data$avg_technology))

### ARIMA Errors
arima_rmse_test_2V <- sqrt(mean((ARIMA_forecast_2V_MT - test_data$avg_technology)^2))
arima_mae_test_2V <- mean(abs(ARIMA_forecast_2V_MT - test_data$avg_technology))


## Print results
cat("\nRF Test Set RMSE:", rmse_rf_tuned_NV, "MAE:", mae_rf_tuned_NV, "\n")
cat("\nARIMA Test Set RMSE:", arima_rmse_test_2V, "MAE:", arima_mae_test_2V)


### Create a data frame with actual values and predictions
comparison_df_2V <- data.frame(
  Date = test_data$date,
  Actual = test_data$avg_technology,
  RF_Predicted = predictions_rf_tuned_NV,
  ARIMA_Predicted = ARIMA_forecast_2V_MT
)

# Truncate the longer vector
min_length <- min(length(test_data$date), 
                  length(test_data$avg_technology), 
                  length(predictions_rf_tuned_NV), 
                  length(ARIMA_forecast_2V_MT))

# Ensure all vectors have the same length
comparison_df_2V <- data.frame(
  Date = test_data$date[1:min_length],
  Actual = test_data$avg_technology[1:min_length],
  RF_Predicted = predictions_rf_tuned_NV[1:min_length],
  ARIMA_Predicted = ARIMA_forecast_2V_MT[1:min_length]
)

ggplot(comparison_df_2V, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1.2) +
  geom_line(aes(y = RF_Predicted, color = "Random Forest"), size = 1.2) +   # Correct column name
  geom_line(aes(y = ARIMA_Predicted, color = "ARIMA"), size = 1.2) +        # Correct column name
  labs(title = "Model Comparison: Random Forest vs ARIMA",
       x = "Date", y = "Technology Stocks") +
  scale_color_manual(values = c("Actual" = "black", "Random Forest" = "blue", "ARIMA" = "red")) +
  theme_minimal()


##### CONCLUSION: RF BETTER ON TEST DATA, ARIMA MIGHT BE OVERFITTING ON TRAIN DATA #####







# Combine actual values and predictions into a single data frame
predictions_df <- data.frame(
  Date = c(train_data$date, test_data$date),  # Dates from both train and test sets
  Actual = c(train_data$avg_technology, test_data$avg_technology),  # Actual values
  Predicted = c(predictions_rf_train, predictions_rf_test),  # Predicted values
  Set = c(rep("Training", nrow(train_data)), rep("Test", nrow(test_data)))  # Indicate if data is from training or test set
)

# Save the predictions dataframe to the specified path
write_xlsx(predictions_df, "C:/Users/Paolo/Desktop/Project/RF_predictions_tech.xlsx")









# ROLLING FORECAST (PREDICT WITH RF)
# Forecasting for multiple steps (e.g., for the next 12 months)
predictions_roll <- numeric(12)  # For 12 months
current_data <- tail(test_data, 1)

# Calculate the average values of the relevant features (e.g., rgdp2017, avg_vix_close) 
# from the training data or earlier data points
avg_rgdp2017 <- mean(train_data$rgdp2017, na.rm = TRUE)  # Average GDP from training data
avg_vix_close <- mean(train_data$avg_vix_close, na.rm = TRUE)  # Average VIX from training data

# Loop to make predictions for each month (or future time period)
for(i in 1:12) {
  # Make prediction for the next month/year
  prediction_roll <- predict(rf_tuned_tech_NV, newdata = current_data)
  
  # Store the predicted value in the predictions_roll vector
  predictions_roll[i] <- prediction_roll
  
  # Update the current_data with the predicted value (e.g., update avg_technology)
  current_data$avg_technology <- prediction_roll
  
  # Update the features for the next period based on averages or trends
  # For example, you can increment `rgdp2017` and `avg_vix_close` by some constant or use a trend model
  
  # Let's assume that both variables will slightly increase each period (just as an example)
  current_data$rgdp2017 <- avg_rgdp2017 * (1 + 0.01)  # Assuming 1% growth per period
  current_data$avg_vix_close <- avg_vix_close * (1 + 0.005)  # Assuming 0.5% growth per period
}

# Print the rolling predictions
print(predictions_roll)








# COMBINATION OF RF AND ARIMA TO MAKE PREDICTIONS

## Make RF predictions for the future using the predicted macroeconomic variables
rf_predictions <- predict(rf_tuned_tech_NV, newdata = data.frame(
  rgdp2017 = future_rgdp_MT,
  avg_vix_close = future_vix_MT
))

combined_predictions <- (ARIMA_forecast_2V_MT + rf_predictions) / 2




# Ensure both vectors have the same length by truncating the longer one
min_length <- min(length(ARIMA_forecast_2V_MT), length(test_data$avg_technology))

stacked_data <- data.frame(
  ARIMA_pred = ARIMA_forecast_2V_MT[1:min_length],
  RF_pred = rf_predictions[1:min_length],
  Actual = test_data$avg_technology[1:min_length]
)

# View the stacked data
head(stacked_data)




## Stack the predictions
stacked_data <- data.frame(
  ARIMA_pred = ARIMA_forecast_2V_MT,
  RF_pred = rf_predictions,
  Actual = test_data$avg_technology  # Actual values for training the meta-model
)

## Fit a meta-model (e.g., linear regression)
meta_model <- lm(Actual ~ ARIMA_pred + RF_pred, data = stacked_data)



# Make predictions from the meta-model
final_predictions <- predict(meta_model, newdata = data.frame(
  ARIMA_pred = ARIMA_forecast_2V_MT,  # Directly use the vector, no need for $mean
  RF_pred = rf_predictions
))

# View the final predictions
head(final_predictions)


## Combine predictions into a dataframe
combined_df <- data.frame(
  Date = seq.Date(from = as.Date("2025-01-01"), by = "quarter", length.out = 20),
  ARIMA = ARIMA_forecast_2V_MT,  # Use the direct numeric vector
  RF = rf_predictions,
  Combined = combined_predictions
)

## Plot
ggplot(combined_df, aes(x = Date)) +
  geom_line(aes(y = ARIMA, color = "ARIMA"), size = 1) +
  geom_line(aes(y = RF, color = "Random Forest"), size = 1) +
  geom_line(aes(y = Combined, color = "Combined"), size = 1, linetype = "dashed") +
  labs(title = "ARIMA and RF Predictions for avg_technology",
       x = "Date", y = "Technology Stocks") +
  scale_color_manual(values = c("ARIMA" = "blue", "Random Forest" = "red", "Combined" = "green")) +
  theme_minimal()

# Ensure the same length before combining
min_length <- min(nrow(combined_df), nrow(test_data))

# Create the final dataframe
final_df <- data.frame(
  Date = combined_df$Date,
  Actual = test_data$avg_technology[1:min_length],   # Match the length
  ARIMA_Predicted = combined_df$ARIMA,
  RF_Predicted = combined_df$RF,
  Combined_Predicted = combined_df$Combined
)

# Save as Excel
write_xlsx(final_df, "C:/Users/Paolo/Desktop/Project/combined_predictions_and_actuals.xlsx")






