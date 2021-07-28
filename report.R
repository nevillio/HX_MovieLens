## ----include=FALSE---------------------------------------------------------------------
knitr::opts_chunk$set(error=FALSE, warning=FALSE, message=FALSE, cache=TRUE, comment = "#")


## ----  echo=FALSE----------------------------------------------------------------------
# Global Ops, Packages, Libraries ####
## Set global options ####
options(repos="https://cran.rstudio.com")
options(timeout=10000, digits=10, pillar.sigfigs=100)
## Install packages ####
list.of.packages <- c("corrplot", "data.table", "ggplot2", "ggthemes", "kableExtra", "knitr", "RColorBrewer", "scales", "tidyverse", "tinytex")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
rm(list.of.packages,new.packages)

## ----library loading-------------------------------------------------------------------
# Load libraries 
library(tidyverse)
library(ggthemes)
library(data.table)
library(corrplot)
library(knitr) #A General-Purpose Package for Dynamic Report Generation in R
library(kableExtra)
library(lubridate)
library(tinytex)
library(latexpdf)
# set global options
options(timeout=10000, digits=5)


## ----file download---------------------------------------------------------------------
#Source file 
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Splitting the data 
# Validation set will be 10% of MovieLens data 
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)


## ----splitting-------------------------------------------------------------------------
# Split data into training and test sets - test set will be 10% of edx 
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(test_index, temp, removed)


## --------------------------------------------------------------------------------------
#Initial exploration
class(edx)
str(edx)
head(edx, 10)


## --------------------------------------------------------------------------------------
unique(edx$rating)
table(edx$rating)
summary(edx$rating)


## --------------------------------------------------------------------------------------
# Data Cleaning - Select the most relevant features
train_set<-train_set%>%select(-timestamp)
test_set<-test_set%>%select(-timestamp)


## --------------------------------------------------------------------------------------
# Define Root Mean Squared Error (RMSE) - loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


## ---- eval=TRUE------------------------------------------------------------------------
# create a results table with all the RMSEs
rmse_results <- tibble(Method = "Project Goal", RMSE = 0.86490)


## --------------------------------------------------------------------------------------
# Mean of observed ratings
mu <- mean(train_set$rating)
mu

# Naive RMSE - predict all unknown ratings with overall mean 
naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse

# Update results table with naive RMSE
rmse_results <- bind_rows(rmse_results, 
                    tibble(Method = "Naive RMSE", 
                           RMSE = RMSE(test_set$rating, mu)))


## --------------------------------------------------------------------------------------
# Calculate movie effect
b_i <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Predict ratings with mean + bi  
y_hat <- mu + test_set %>% 
  left_join(b_i, by = "movieId") %>% 
  .$b_i
RMSE(test_set$rating, y_hat)

# Update results table with movie effect RMSE
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Movie Effect", 
                                 RMSE = RMSE(test_set$rating, y_hat)))


## --------------------------------------------------------------------------------------
# Calculate user effect
b_u <- train_set %>% 
  left_join(b_i, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict ratings with mean + bi + bu
y_hat <- test_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
RMSE(test_set$rating, y_hat)

# Update results table with movie effect RMSE
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Movie + User Effect", 
                                 RMSE = RMSE(test_set$rating, y_hat)))


## ----genre effect----------------------------------------------------------------------
# Calculate genres effect
b_g <- train_set %>%
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        group_by(genres) %>% 
        summarise(b_g = mean(rating - b_i - b_u -mu))

# Predict ratings 
y_hat <- test_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
RMSE(test_set$rating, y_hat)

# Update results table with movie effect RMSE
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Movie + User + Genres Effect", 
                                 RMSE = RMSE(test_set$rating, y_hat)))


## --------------------------------------------------------------------------------------
# Evaluate Model Results
# calculate biggest residuals (errors)
test_set %>% 
  left_join(b_i, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%  
  head(10)


## --------------------------------------------------------------------------------------
# create a database of movie titles
movie_titles <- train_set %>% 
  select(movieId, title) %>%
  distinct()


## --------------------------------------------------------------------------------------
# 10 best movies according to b_i 
b_i %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  head(10)  %>% 
  select(title)


## --------------------------------------------------------------------------------------
# 10 worst movies according to bi 
b_i %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  head(10)  %>% 
  select(title)


## ----regularization--------------------------------------------------------------------
# Define a set of lambdas to tune
lambdas <- seq(0, 10, 0.25)
# Tune the lambdas using regularization function
regularization <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
 
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- train_set %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>% 
    summarise(b_g = sum(rating - b_i - b_u -mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})
# Plot - lambdas vs RMSE
qplot(lambdas, regularization)  
# Choose the lambda which produces the lowest RMSE
lambda<- lambdas[which.min(regularization)]
lambda


## --------------------------------------------------------------------------------------
# Calculate the movie and user effects with the best lambda (parameter) 
mu <- mean(train_set$rating)

# Movie effect (bi)
bi_reg <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# User effect (bu)
bu_reg <- train_set %>% 
  left_join(bi_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Genres Effect
bg_reg <- train_set %>%
  left_join(b_i, by = 'movieId') %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>% 
  summarise(b_g = sum(rating - b_i - b_u -mu)/(n()+lambda))

# Prediction with regularized bi and bu 
y_hat_reg <- test_set %>% 
  left_join(bi_reg, by = "movieId") %>%
  left_join(bu_reg, by = "userId") %>%
  left_join(bg_reg, by = "genres") %>% 
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
# Update results table with regularized movie + user effect
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Regularized Model", 
                                 RMSE = RMSE(test_set$rating, y_hat_reg)))


## --------------------------------------------------------------------------------------
# Top 10 best movies after regularization
test_set %>% 
  left_join(bi_reg, by = "movieId") %>%
  left_join(bu_reg, by = "userId") %>% 
  left_join(bg_reg, by = "genres") %>% 
  mutate(pred = mu + b_i + b_u + b_g) %>% 
  arrange(desc(pred)) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)


## --------------------------------------------------------------------------------------
# Top 10 worst movies after regularization
test_set %>% 
  left_join(bi_reg, by = "movieId") %>%
  left_join(bu_reg, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  arrange(pred) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)


## --------------------------------------------------------------------------------------
# Calculate the movie and user effects with the best lambda (parameter) 
mu <- mean(edx$rating)

# Movie effect (bi)
bi_fin <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# User effect (bu)
bu_fin <- edx %>% 
  left_join(bi_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Genres Effect
bg_fin <- edx %>%
  left_join(b_i, by = 'movieId') %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>% 
  summarise(b_g = sum(rating - b_i - b_u -mu)/(n()+lambda))

# Prediction with regularized bi and bu 
y_hat_fin <- validation %>% 
  left_join(bi_reg, by = "movieId") %>%
  left_join(bu_reg, by = "userId") %>%
  left_join(bg_reg, by = "genres") %>% 
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
# Update results table with regularized movie + user effect
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Final Model", 
                                 RMSE = RMSE(validation$rating, y_hat_fin)))


## --------------------------------------------------------------------------------------
knitr::kable(rmse_results)%>% kable_styling()


## --------------------------------------------------------------------------------------
# top 10 best movies predicted by matrix factorization
validation %>% 
  mutate(pred = y_hat_fin) %>% 
  arrange(desc(pred)) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)
# top 10 worst movies predicted by matrix factorization
validation %>% 
  mutate(pred = y_hat_fin) %>% 
  arrange(pred) %>% 
  group_by(title) %>% 
  select(title) %>%
  head(10)

