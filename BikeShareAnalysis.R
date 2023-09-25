library(tidyverse)
library(vroom)
library(tidymodels)
library(poissonreg)
library(glmnet)
library(randomForest)
library(xgboost)

bike <- vroom("./STAT\ 348/KaggleBikeShare/train.csv")
bike

bike_clean <- bike %>% 
  select(-c('casual', 'registered'))
bike_clean

test <- vroom("./STAT\ 348/KaggleBikeShare/test.csv")
test

# cleaning by changing the value of 4 for the weather column to 3

## just one instance of weather == 4
bike_clean %>%
  filter(bike["weather"] == 4)

## changing the value of weather == 4 to 3
bike_clean[bike_clean$weather == 4, ]["weather"] <- 3

## no weather values of 4
ggplot(bike_clean, aes(weather, count)) +
  geom_point()

bike_clean <- bike_clean %>%
  mutate(count = log(count))

# feature engineering
## turn holiday column to a factor and remove the holiday column
## turn workingday column to a factor and remove the workingday column
## turn weather column to a factor and remove the weather column
## decode numbers for season by the respective season name
## turn season column to a factor and remove season column
bike_recipe <- recipe(count~., data=bike_clean) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped_bike_recipe <- prep(bike_recipe)
bake(prepped_bike_recipe, new_data=bike_clean)
bake(prepped_bike_recipe, new_data=test)

model <- linear_reg(penalty = .05, mixture = .1) %>%
  set_engine("glmnet")

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(model) %>%
  fit(data = bike_clean)

bike_predictions <- predict(bike_workflow,
                            new_data = test)
bike_predictions["datetime"] <- test["datetime"]
bike_predictions <- bike_predictions %>%
  select(c(datetime, .pred)) %>%
  mutate(.pred=exp(.pred))
names(bike_predictions) <- c("datetime", "count")
bike_predictions

bike_predictions_data_frame <- data.frame(bike_predictions)
bike_predictions_data_frame$count[bike_predictions_data_frame$count < 0] <- 0
bike_predictions_data_frame$count[is.na(bike_predictions_data_frame$count)] <- 0
bike_predictions_data_frame

extract_fit_engine(bike_workflow) %>%
  summary()

write.csv(bike_predictions_data_frame, "./STAT\ 348/KaggleBikeShare/predictions.csv", row.names = F)

# Poisson regression

# pois_mod <- poisson_reg() %>%
#   set_engine("glm")
# 
# bike_pois_workflow <- workflow() %>%
# add_recipe(bike_recipe) %>%
# add_model(pois_mod) %>%
# fit(data = bike_clean) # Fit the workflow
# 
# bike_predictions_poisson <- predict(bike_pois_workflow,
#                             new_data=test) # Use fit to predict
# bike_predictions_poisson["datetime"] <- test["datetime"]
# bike_predictions_poisson <- bike_predictions_poisson %>%
#   select(c(datetime, .pred))
# names(bike_predictions_poisson) <- c("datetime", "count")
# 
# bike_predictions_poisson_data_frame <- data.frame(bike_predictions_poisson)
# bike_predictions_poisson_data_frame$count[bike_predictions_poisson_data_frame$count < 0] <- 0
# bike_predictions_poisson_data_frame$count[is.na(bike_predictions_poisson_data_frame$count)] <- 0
# bike_predictions_poisson_data_frame
# 
# write.csv(bike_predictions_poisson_data_frame, "./STAT\ 348/KaggleBikeShare/predictions_poisson.csv", row.names = F)

xgb_spec <- boost_tree(mode = "regression") %>%
  set_engine("xgboost")

# Define a workflow
xgb_workflow <- workflow(count~.) %>%
  add_recipe(bike_recipe) %>%
  add_model(xgb_spec)

# Fit the model
xgb_fit <- xgb_workflow %>%
  fit(data = bike_clean)

xgb_fit

predictions <- predict(xgb_fit, test)
predictions["datetime"] <- test["datetime"]
predictions <- predictions %>%
  select(c(datetime, .pred)) %>%
  mutate(.pred=exp(.pred))
names(predictions) <- c("datetime", "count")
predictions

bike_predictions_data_frame <- data.frame(predictions)
bike_predictions_data_frame$count[bike_predictions_data_frame$count < 0] <- 0
bike_predictions_data_frame$count[is.na(bike_predictions_data_frame$count)] <- 0
bike_predictions_data_frame
write.csv(bike_predictions_data_frame, "./STAT\ 348/KaggleBikeShare/xgboost_predictions.csv", row.names = F)

# 9/25 tune and fit penalized regression model
tuning_penalized_regression_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
add_recipe(bike_recipe) %>%
add_model(tuning_penalized_regression_model)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(bike_clean, v = 5, repeats=1)

## Run the CV
CV_results <- preg_wf %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(rmse, mae)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best("rmse")

model <- linear_reg(penalty = .0000000001, mixture = .25) %>%
  set_engine("glmnet")

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(model) %>%
  fit(data = bike_clean)

bike_predictions <- predict(bike_workflow,
                            new_data = test)
bike_predictions["datetime"] <- test["datetime"]
bike_predictions <- bike_predictions %>%
  select(c(datetime, .pred)) %>%
  mutate(.pred=exp(.pred))
names(bike_predictions) <- c("datetime", "count")

bike_predictions_data_frame <- data.frame(bike_predictions)
bike_predictions_data_frame$count[bike_predictions_data_frame$count < 0] <- 0
bike_predictions_data_frame$count[is.na(bike_predictions_data_frame$count)] <- 0
bike_predictions_data_frame

write.csv(bike_predictions_data_frame, "./STAT\ 348/KaggleBikeShare/tuning_predictions.csv", row.names = F)
