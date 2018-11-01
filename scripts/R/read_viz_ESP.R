# Visualizing streamflow forecasts: official and ESP (ensemble forecast predictions)
# author: Billy Raseman

```{r}
# clear environment
rm(list=ls())

# load packages
library(tidyverse)

# read in data
locations <- c("MPHC2_", "CAMC2_", "DOLC2_")
nlocs <- length(locations)
# file.end <- c("2013", "2014", "2015", "2016", 
              # "2017", "2018", "historical")
years <- c("2013", "2014", "2015", "2016", "2017", "2018")
nfiles <- length(years)
# files <- file.paths <- vector(mode="character", length=nfiles)
files <- file.paths <- c()
# i <- 1

# list of dataframes
df.list <- list()

n.col <- c()
for (i in 1:nfiles) {
  for (j in 1:nlocs) {
# i <- 3
  files[i] <- str_c(locations[j], years[i], ".csv")
  data.dir <- "../../data/"
  file.paths[i] <- str_c(data.dir, files[i])
  
  temp.df <- read_csv(file=file.paths[i])
  
  n.col[i] <- ncol(temp.df)  # get number of columns for debugging

  if (ncol(temp.df) == 11) {
    colnames(temp.df) <- c("date", "average", "obs_accum", "obs_total",
                           "normal_accum", "esp_10", "esp_50", "esp_90",
                           "esp_10_wo_obs", "esp_50_wo_obs", "esp_90_wo_obs")
  } else if (ncol(temp.df) == 19) {
    colnames(temp.df) <- c("date", "average", "obs_accum", "obs_total",
                           "normal_accum", "esp_max", "esp_10", "esp_30", 
                           "esp_50", "esp_70", "esp_90", "esp_min", 
                           "esp_max_wo_obs", "esp_10_wo_obs", "esp_30_wo_obs", 
                           "esp_50_wo_obs", "esp_70_wo_obs", "esp_90_wo_obs",
                           "esp_min_wo_obs")
  } else {
    print("Wrong number of columns!")
  }

  temp2.df <- select(temp.df, date, esp_10, esp_50, esp_90, obs_total) %>%
    mutate(year = as.factor(years[i]), 
           location = as.factor(locations[j]))
  
  if (i == 1) {
    df <- temp2.df
  } else {
    df <- rbind(df, temp2.df)
  }
  }
}
# 
# # visualize the three datasets
# ggplot(data=df, aes(x=date, color=year)) +
#   # geom_line(aes(y=esp_90, linetype="dashed")) +
#   # geom_line(aes(y=esp_10, linetype="dotted")) +
#   geom_line(aes(y=obs_total), linetype="dashed") +
#   geom_line(aes(y=esp_50)) +
#   facet_grid(location ~ ., scales="free_y") +
#   ylab("volume (kiloacre-ft)") 
```

# visualize just the Dolores
ggplot(data=filter(df,location==locations[3]), aes(x=date, color=year)) +
  # geom_line(aes(y=esp_90, linetype="dashed")) +
  # geom_line(aes(y=esp_10, linetype="dotted")) +
  geom_line(aes(y=obs_total), linetype="dashed") +
  geom_line(aes(y=esp_50), linetype="solid") +
  ylab("volume (kiloacre-ft)") +
  ggtitle("Dolores River (dashed=observed, solid=forecast)")

# normalize these data 
# df <- mutate(df, esp_50_scaled = scale(esp_50))
# 
# ggplot(data=df, aes(x=date, color=location)) +
#   # geom_line(aes(y=esp_90)) +
#   # geom_line(aes(y=esp_10)) +
#   geom_line(aes(y=esp_50_scaled)) 
