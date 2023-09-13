# Bike Share EDA Code

library(tidyverse)
library(vroom)
library(DataExplorer)
library(patchwork)

# Read in the data
bike <- vroom("./STAT\ 348/KaggleBikeShare/train.csv")
bike

## 2

plot_intro(bike)
dplyr::glimpse(bike)

# count vs. registered, count vs. casual are highly correlated
DataExplorer::plot_correlation(bike)

DataExplorer::plot_bar(bike)

# registered and casual and count are right skewed
DataExplorer::plot_histogram(bike)

# no missing data
DataExplorer::plot_missing(bike)

## 3
count_vs_registered <- ggplot(bike, mapping = aes(registered, count)) +
  geom_point()

count_vs_casual <- ggplot(bike, mapping = aes(casual, count)) +
  geom_point()

registered_bar <- ggplot(bike, mapping = aes(temp)) +
  geom_bar()

humidity_bar <- ggplot(bike, mapping = aes(humidity)) +
  geom_bar()

multiple_plots <- (count_vs_registered + count_vs_casual) / (registered_bar + humidity_bar)

ggsave("four_plots.png", plot = multiple_plots, path = "./STAT\ 348/KaggleBikeShare")
