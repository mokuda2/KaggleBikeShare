library(tidyverse)
library(vroom)
library(tidymodels)
library(poissonreg)
library(glmnet)
library(randomForest)
library(xgboost)
library(stacks)

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

bike_clean <- bike_clean %>%
  mutate(count = log(count))

# feature engineering
## turn holiday column to a factor and remove the holiday column
## turn workingday column to a factor and remove the workingday column
## turn weather column to a factor and remove the weather column
## decode numbers for season by the respective season name
## turn season column to a factor and remove season column
# turn year into factor
bike_recipe <- recipe(count~., data=bike_clean) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_date(datetime, features="year") %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime) %>%
  step_mutate(datetime_hour=factor(datetime_hour, levels=0:23, labels=0:23)) %>%
  step_mutate(datetime_year=factor(datetime_year, levels=c(2011, 2012), labels=c("2011", "2012"))) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped_bike_recipe <- prep(bike_recipe)
bake(prepped_bike_recipe, new_data=bike_clean)
bake(prepped_bike_recipe, new_data=test)

# Define the XGBoost model
xgb_model <- boost_tree(
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  tree_depth = tune()
) %>%
  set_mode("regression") %>%
  set_engine("xgboost", nthread = parallel::detectCores())

# Create the workflow
bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(xgb_model)

# Fit the model
hyper_grid <- grid_regular(
  mtry(range = c(1, ncol(bike_clean) - 1)), 
  trees(range = c(50, 150)), 
  min_n(range = c(5, 20)), 
  tree_depth(range = c(3, 8))
)

bike_fit <- bike_workflow %>%
  tune_grid(
    resamples = bootstraps(bike_clean, strata = NULL),
    grid = hyper_grid,
    metrics = metric_set(rmse)
  )

# Extract the best model
best_model <- bike_fit %>%
  select_best("rmse")

final_wf <- bike_workflow %>%
  finalize_workflow(best_model) %>%
  fit(data=bike_clean)

## Predict
predict_final_wf <- final_wf %>%
  predict(new_data = test)

predict_final_wf["datetime"] <- test["datetime"]
predict_final_wf <- predict_final_wf %>%
  select(c(datetime, .pred)) %>%
  mutate(.pred=exp(.pred))
names(predict_final_wf) <- c("datetime", "count")

bike_predictions_data_frame <- data.frame(predict_final_wf)
bike_predictions_data_frame$count[bike_predictions_data_frame$count < 0] <- 0
bike_predictions_data_frame$count[is.na(bike_predictions_data_frame$count)] <- 0
bike_predictions_data_frame

write.csv(bike_predictions_data_frame, "./STAT\ 348/KaggleBikeShare/xgboost.csv", row.names = F)

###############################################################################
## regression model

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

final_wf <- preg_wf %>%
finalize_workflow(bestTune) %>%
fit(data=bike_clean)

## Predict
predict_final_wf <- final_wf %>%
predict(new_data = test)

predict_final_wf["datetime"] <- bike_predictions["datetime"]
predict_final_wf <- predict_final_wf %>%
  select(c(datetime, .pred)) %>%
  mutate(.pred=exp(.pred))
names(predict_final_wf) <- c("datetime", "count")

bike_predictions_data_frame <- data.frame(predict_final_wf)
bike_predictions_data_frame$count[bike_predictions_data_frame$count < 0] <- 0
bike_predictions_data_frame$count[is.na(bike_predictions_data_frame$count)] <- 0
bike_predictions_data_frame

write.csv(bike_predictions_data_frame, "./STAT\ 348/KaggleBikeShare/tuning_predictions.csv", row.names = F)

## regression trees
my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use7
  set_mode("regression")

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

preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod)

tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 3)

folds <- vfold_cv(bike_clean, v = 5, repeats=1)

# Run the CV
CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae))
bestTune <- CV_results %>% select_best('rmse')

final_model <- finalize_workflow(preg_wf,bestTune) %>%
  fit(data=bike_clean)

bike_predictions <- predict(final_model,
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

write.csv(bike_predictions_data_frame, "./STAT\ 348/KaggleBikeShare/regression_trees.csv", row.names = F)

## random forest
random_forest_mod <- rand_forest(mtry = tune(),
                        min_n=tune(),
                        trees=1000) %>% #Type of model
  set_engine("ranger") %>% # Engine = What R function to use7
  set_mode("regression")

rf_bike_recipe <- recipe(count~., data=bike_clean) %>%
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

rf_workflow <- workflow() %>%
  add_recipe(rf_bike_recipe) %>%
  add_model(random_forest_mod)

tuning_grid <- grid_regular(mtry(range=c(1,(ncol(bike_clean) - 1))),
                            min_n(),
                            levels = 3)

folds <- vfold_cv(bike_clean, v = 5, repeats=1)

# Run the CV
CV_results <- rf_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae))

bestTune <- CV_results %>% select_best('rmse')

final_model <- finalize_workflow(rf_workflow, bestTune) %>%
  fit(data=bike_clean)

bike_predictions <- predict(final_model,
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

write.csv(bike_predictions_data_frame, "./STAT\ 348/KaggleBikeShare/random_forest.csv", row.names = F)

## stacking

# recipe
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

# cv with vfold_cv
folds <- vfold_cv(bike_clean, v = 10, repeats=1)

# tuned and untuned
untunedModel <- control_stack_grid() # grid of parameters
tunedModel <- control_stack_resamples() # tuned

# define linear model and then workflow
# fit linear regression to folds
lin_reg_model <- linear_reg() %>%
  set_engine("lm")

linreg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_reg_model)

linreg_folds_fit <- linreg_wf %>%
  fit_resamples(resamples=folds,
                control=tunedModel)

# penalized regression
# tune reg model and then workflow
tuning_penalized_regression_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(tuning_penalized_regression_model)

# set up tuning grid
penReg_tuneGrid <- grid_regular(mixture(),
                                penalty(),
                                levels=3)

# fit to folds
penReg_folds_fit <- preg_wf %>%
  tune_grid(resamples=folds,
                grid=penReg_tuneGrid,
                metrics=metric_set(rmse),
                control=untunedModel)

# set up regression tree
regression_tree_model <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use7
  set_mode("regression")

regTree_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(regression_tree_model)

regTree_tuneGrid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 3)

tree_folds_fit <- regTree_wf %>%
  tune_grid(resamples=folds,
            grid=regTree_tuneGrid,
            metrics=metric_set(rmse),
            control=untunedModel)

# random forest
random_forest_mod <- rand_forest(mtry = tune(),
                                 min_n=tune(),
                                 trees=1000) %>% #Type of model
  set_engine("ranger") %>% # Engine = What R function to use7
  set_mode("regression")

rf_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(random_forest_mod)

rf_tuning_grid <- grid_regular(mtry(range=c(1,(ncol(bike_clean) - 1))),
                            min_n(),
                            levels = 5)

forest_folds_fit <- rf_workflow %>%
  tune_grid(resamples=folds,
            grid=rf_tuning_grid,
            metrics=metric_set(rmse),
            control=untunedModel)

# stack models
bike_stack <- stacks() %>%
  add_candidates(linreg_folds_fit) %>%
  add_candidates(penReg_folds_fit) %>% # give all 25
  add_candidates(tree_folds_fit) %>%
  add_candidates(forest_folds_fit)

as_tibble(bike_stack)

fitted_bike_stack <- bike_stack %>%
  blend_predictions() %>%
  fit_members()

collect_parameters(fitted_bike_stack, "tree_folds_fit")

# predictions
bike_predictions <- predict(fitted_bike_stack,
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

write.csv(bike_predictions_data_frame, "./STAT\ 348/KaggleBikeShare/stacking.csv", row.names = F)
