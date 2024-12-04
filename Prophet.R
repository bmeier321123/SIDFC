library(vroom)
library(modeltime) #Extensions of tidymodels to time series1
library(timetk)
library(prophet)
library(tidyverse)
library(dplyr)
library(tidymodels)
library(ggplot2)
library(prophet)

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


prophet_model1 <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split1))

prophet_model2 <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split2))



## Calibrate (tune) the models (find p,d,q,P,D,Q)
cv_results1 <- modeltime_calibrate(prophet_model1,
                                   new_data = testing(cv_split1))

cv_results2 <- modeltime_calibrate(prophet_model2,
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

fullfit1 <- cv_results1 %>%
  modeltime_refit(data=storeItemTrain1)

preds <- fullfit1 %>% 
  modeltime_forecast(
    new_data = storeItemTest1,
    actual_data = storeItemTrain1
  ) %>% 
  filter(!is.na(.model_id)) %>%
  mutate(id=storeItemTest1$id) %>%
  select(id, .value) %>%
  rename(sales=.value)
  
  

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
