# Bike Share EDA Code

library(tidyverse)
library(vroom)
library(DataExplorer)

# Read in the data
bike <- vroom("./STAT\ 348/KaggleBikeShare/train.csv")
bike

plot_intro(bike)
dplyr::glimpse(bike)
DataExplorer::plot_correlation(bike)
DataExplorer::plot_bar(bike)

# registered and casual and count are right skewed
DataExplorer::plot_histogram(bike)

# no missing data
DataExplorer::plot_missing(bike)
