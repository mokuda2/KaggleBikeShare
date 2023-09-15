library(tidyverse)
library(vroom)
library(tidymodels)

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

# feature engineering
## turn holiday column to a factor and remove the holiday column
## turn workingday column to a factor and remove the workingday column
## turn weather column to a factor and remove the weather column
## decode numbers for season by the respective season name
## turn season column to a factor and remove season column
my_bike_recipe <- recipe(count ~ ., data=bike_clean) %>%
  step_mutate(holiday, factor(holiday)) %>%
  step_rm(holiday) %>%
  step_mutate(weather, factor(weather)) %>%
  step_rm(weather) %>%
  step_mutate(workingday, factor(workingday)) %>%
  step_rm(workingday) %>%
  step_mutate(season_string = 
                recode(season, `1` = "spring", 
                       `2` = "summer", 
                       `3` = "fall", 
                       `4` = "winter")) %>%
  step_mutate(season, factor(season)) %>%
  step_rm(season)

prepped_bike_recipe <- prep(my_bike_recipe)
bake(prepped_bike_recipe, new_data=bike_clean)

model <- linear_reg() %>%
  set_engine("lm")

bike_workflow <- workflow() %>%
  add_recipe(my_bike_recipe) %>%
  add_model(model) %>%
  fit(data = bike_clean)

bike_predictions <- predict(bike_workflow,
                            new_data = test)
bike_predictions["datetime"] <- test["datetime"]


bike_predictions_data_frame <- data.frame(bike_predictions)
bike_predictions_data_frame$count[bike_predictions_data_frame$count < 0] <- 0
bike_predictions_data_frame$count[is.na(bike_predictions_data_frame$count)] <- 0
bike_predictions_data_frame

vroom_write(bike_predictions_data_frame, "./STAT\ 348/KaggleBikeShare/predictions.csv")
