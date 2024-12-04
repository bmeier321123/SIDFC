library(vroom)
library(modeltime) #Extensions of tidymodels to time series1
library(timetk)
library(prophet)
library(tidyverse)
library(dplyr)
library(tidymodels)
library(ggplot2)

# Read in Data----------------------------
trainData <- vroom("train.csv")
testData <- vroom("test.csv")


## Read in the Data and filter to store/item
storeItemTrain1 <- trainData %>%
  filter(store==7, item==27)

storeItemTrain2 <- trainData %>%
  filter(store==9, item==48)

storeItemTest1 <- testData %>%
filter(store==7, item==27)

storeItemTest2 <- testData %>%
  filter(store==9, item==48)


## Create the CV split for time series
cv_split1 <- time_series_split(storeItemTrain1, assess="3 months", cumulative = TRUE)
cv_split2 <- time_series_split(storeItemTrain2, assess="3 months", cumulative = TRUE)

## Create a recipe for the linear model part
arima_recipe1 <- recipe(sales ~ ., data = training(cv_split1)) %>% 
  step_date( date, features = "dow") %>% 
  step_date(date, features = "month") %>% 
  step_mutate_at(date_dow, fn = factor) %>% 
  step_mutate_at(date_month, fn = factor) %>%
  step_date(date, features = "decimal") %>% 
  step_mutate(date_decimal = as.numeric(date_decimal)) %>% 
  step_zv(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors())

arima_recipe2 <- recipe(sales ~ ., data = training(cv_split2)) %>% 
  step_date( date, features = "dow") %>% 
  step_date(date, features = "month") %>% 
  step_mutate_at(date_dow, fn = factor) %>% 
  step_mutate_at(date_month, fn = factor) %>%
  step_date(date, features = "decimal") %>% 
  step_mutate(date_decimal = as.numeric(date_decimal)) %>% 
  step_zv(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors())

## Define the ARIMA Model
arima_model <- arima_reg(seasonal_period=365,
                         non_seasonal_ar=5, # default max p to tune
                         non_seasonal_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2 #default max D to tune
) %>%
set_engine("auto_arima")

## Merge into a single workflow and fit to the training data
arima_wf1 <- workflow() %>%
add_recipe(arima_recipe1) %>%
add_model(arima_model) %>%
fit(data=training(cv_split1))

arima_wf2 <- workflow() %>%
  add_recipe(arima_recipe2) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split2))

## Calibrate (tune) the models (find p,d,q,P,D,Q)
cv_results1 <- modeltime_calibrate(arima_wf1,
                                  new_data = testing(cv_split1))

cv_results2 <- modeltime_calibrate(arima_wf2,
                                  new_data = testing(cv_split2))
## Visualize results
cv_plot1 <- cv_results1 %>% 
  modeltime_forecast(
    new_data = testing(cv_split1),
    actual_data = training(cv_split1)
  ) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

cv_plot2 <- cv_results2 %>% 
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = training(cv_split2)
  ) %>% 
  plot_modeltime_forecast(.interactive = FALSE)



## Now that you have calibrated (tuned) refit to whole dataset
fullfit1 <- cv_results1 %>%
modeltime_refit(data=storeItemTrain1)

forecast1 <- fullfit1 %>% 
  modeltime_forecast(
    new_data = storeItemTest1,
    actual_data = storeItemTrain1
  ) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

fullfit2 <- cv_results2 %>%
  modeltime_refit(data=storeItemTrain2)

forecast2 <- fullfit2 %>% 
  modeltime_forecast(
    new_data = storeItemTest2,
    actual_data = storeItemTrain2
  ) %>% 
  plot_modeltime_forecast(.interactive = FALSE)


plotly::subplot(cv_plot1, cv_plot2, forecast1, forecast2, nrows = 2)
