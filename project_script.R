# Libraries -----------------------------------------------------------------------------------

library(dplyr)
library(glue)
library(ggplot2)
library(randomForest)
library(tree)
library(pROC)
library(caret)
library(e1071)
library(stringr)
library(ROSE)
library(caret)

# Reading the csv -----------------------------------------------------------------------------

tweet_df <- read.csv(file = "tweets.csv")
tweet_df <- tweet_df %>% subset(select = -X)

tweet_df$created_at   <- as.factor(tweet_df$created_at)
tweet_df$source       <- as.factor(tweet_df$source)
tweet_df$media_type   <- as.factor(tweet_df$media_type)
tweet_df$lang         <- as.factor(tweet_df$lang)
tweet_df$has_url      <- as.factor(tweet_df$has_url)
tweet_df$has_mentions <- as.factor(tweet_df$has_mentions)
tweet_df$class        <- as.factor(tweet_df$class)
tweet_df$len_hashtags <- as.integer(tweet_df$len_hashtags)

# Data plotting -------------------------------------------------------------------------------

plot_df <- tweet_df

populars_account <- plot_df[plot_df$followers_count > 500,]

sp <- plot_df[plot_df$class == "popular",] %>% ggplot(aes(fill = verified,class)) + geom_bar(position="dodge")
sp

sp <- plot_df %>% ggplot(aes(fill = media_type,class)) + geom_bar(position="dodge")
sp

sp <- plot_df %>% ggplot(aes(fill = source,class)) + geom_bar(position="dodge")
sp

sp <- plot_df[plot_df$class == "popular",] %>% ggplot(aes(fill = created_at,class)) + geom_bar(position="dodge")
sp

sp <- plot_df[plot_df$class == "popular",] %>% ggplot(aes(statuses_count)) + geom_histogram()
sp

sp <- plot_df[plot_df$class == "popular" & plot_df$followers_count > 500,] %>% ggplot(aes(followers_count)) + geom_histogram()
sp

#--------------------------------------------------------------------------------------------------
# Training and Test set ---------------------------------------------------------------------------

train <- sample(1:nrow(tweet_df), as.integer(nrow(tweet_df)*0.8))
tweet_df.test <- tweet_df[-train,]
summary(tweet_df[train,]$class)
summary(tweet_df.test$class)
popular.test <- tweet_df$class[-train]

# Learning techniques------------------------------------------------------------------------------

# Tree ----------------------------------------------------

tr <- tree(formula = class ~ . -screen_name -text -lang -favorite_count -retweet_count -mean -sd -threshold, data = tweet_df[train,],  na.action=na.exclude, subset = train)
pred <- predict(tr, tweet_df.test, type = "class")
table(pred, popular.test)
plot(tr)
text(tr)

# Random Forest ------------------------------------------

# Over-sampling the less numerous class (popular)

over <- ovun.sample(class ~  . -screen_name -text -favorite_count -retweet_count -mean -sd -threshold, data = tweet_df[train,],
                    method = "over", N = 7134*2)$data

# Over-sampling the less numerous class (popular) and under-sampling the most numerous one (not popular)

both <- ovun.sample(class ~  . -screen_name -text -favorite_count -retweet_count -mean -sd -threshold, data = tweet_df[train,],
                    method = "both", p = 0.5, N = 10000)$data

# Random Forest with imbalanced data

rf <- randomForest(formula = class ~ . -screen_name -text -favorite_count -retweet_count -mean -sd -threshold, data = tweet_df[train,],  na.action=na.exclude)
pred <- predict(rf, tweet_df.test, type = "class")

importance(rf)

confusionMatrix(pred, tweet_df.test$class, positive = "popular")

# Random Forest with over-sampled data (balanced dataset)

rf <- randomForest(formula = class ~ . -screen_name -text -favorite_count -retweet_count -mean -sd -threshold, data = over,  na.action=na.exclude)
pred <- predict(rf, tweet_df.test, type = "class")

importance(rf)

confusionMatrix(pred, tweet_df.test$class, positive = "popular")

# Random Forest with both under-sampled and over-sampled data (balanced dataset)

rf <- randomForest(formula = class ~ . -screen_name -text -favorite_count -retweet_count -mean -sd -threshold, data = both,  na.action=na.exclude)
pred <- predict(rf, tweet_df.test, type = "class")

importance(rf)

confusionMatrix(pred, tweet_df.test$class, positive = "popular")

# ROC curve of Random Forest with balanced data

pred <- predict(rf, tweet_df.test, type = "prob")

par(pty = "s")
roc(tweet_df.test$class, pred[,1], plot=TRUE, legacy.axes = TRUE,
    xlab = "False Positive Rate", ylab = "True Positive Rate",
    col = "#377eb8", lwd = 4, print.auc = TRUE)

# Bagging ------------------------------------------------

# Imbalanced data

bag <- randomForest(formula = class ~ . -screen_name -text -favorite_count -retweet_count -mean -sd -threshold, data = tweet_df[train,],  na.action=na.exclude, mtry = 13)
pred <- predict(bag, tweet_df.test, type = "class")

importance(bag)

confusionMatrix(pred, tweet_df.test$class, positive = "popular")

# Over-sampled data (balanced dataset)

bag <- randomForest(formula = class ~ . -screen_name -text -favorite_count -retweet_count -mean -sd -threshold, data = over,  na.action=na.exclude, mtry = 13)
pred <- predict(bag, tweet_df.test, type = "class")

importance(bag)

confusionMatrix(pred, tweet_df.test$class, positive = "popular")

# Over-sampled and under-sampled data (balanced dataset)

bag <- randomForest(formula = class ~ . -screen_name -text -favorite_count -retweet_count -mean -sd -threshold, data = both,  na.action=na.exclude, mtry = 13)
pred <- predict(bag, tweet_df.test, type = "class")

importance(bag)

confusionMatrix(pred, tweet_df.test$class, positive = "popular")

pred <- predict(bag, tweet_df.test, type = "prob")

par(pty = "s")
roc(tweet_df.test$class, pred[,1], plot=TRUE, legacy.axes = TRUE,
    col = "#4daf4a", lwd = 4, add=TRUE, print.auc = TRUE, print.auc.y = 0.4)

legend("bottomright", legend = c("Random Forest", "Bagging"), col = c("#377eb8", "#4daf4a"), lwd = 4)
summary(tweet_df[tweet_df$class == "popular","len_hashtags"])

# Tuning -----------------------------------------------------------------------

# The Random Forest technique has been selected. Now, we tune the model using OOB errors to select the best mtry to use in the training phase.

rff <- tuneRF(both[,c(-1,-3,-5,-6,-20,-21)], both[,"class"], ntreeTry = 500, stepFactor = 0.5, doBest = TRUE)

rf <- randomForest(formula = class ~ . -screen_name -text -favorite_count -retweet_count -threshold, data = both,
                   na.action=na.exclude, mtry=6)

pred <- predict(rf, tweet_df.test, type = "class")
confusionMatrix(pred, tweet_df.test$class, positive = "popular")

pred <- predict(rf, tweet_df.test, type = "prob")
par(pty = "s")
roc <- roc(tweet_df.test$class, pred[,1], plot=TRUE, legacy.axes = TRUE,
           xlab = "False Positive Rate", ylab = "True Positive Rate",
           col = "#377eb8", lwd = 4, print.auc = TRUE)