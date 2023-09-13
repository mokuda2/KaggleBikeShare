library(tidyverse)
library(vroom)
library(tidymodels)

bike <- vroom("./STAT\ 348/KaggleBikeShare/train.csv")
bike

# cleaning by changing the value of 4 for the weather column to 3

## just one instance of weather == 4
bike %>%
  filter(bike["weather"] == 4)

## changing the value of weather == 4 to 3
bike[bike$weather == 4, ]["weather"] <- 3

## no weather values of 4
ggplot(bike, aes(weather, count)) +
  geom_point()

# feature engineering
## turn holiday column to a factor and remove the holiday column
## turn workingday column to a factor and remove the workingday column
## decode numbers for season by the respective season name
## turn season column to a factor and remove season column
my_bike_recipe <- recipe(count ~ ., data=bike) %>%
  step_mutate(holiday, factor(holiday)) %>%
  step_rm(holiday) %>%
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
bake(prepped_bike_recipe, new_data=bike)

