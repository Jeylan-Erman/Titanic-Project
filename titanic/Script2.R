
dir <- "/Users/Jeylan/Documents/Titanic-Project/titanic/"
setwd(dir)

test <- read.csv("test.csv", header=T, as.is=F, na.strings=c("","NA"))
train <- read.csv("train.csv", header=T, as.is=F, na.strings=c("","NA")) 

install.packages("rpart.plot")	

library(corrplot)
library(ggplot2)
library(scales)
library(pROC)
library(car)
library(fields)
library(caret)
library(e1071)
library(dplyr)
library(plyr)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)

train$set <- "train"
test$set  <- "test"
test$Survived <- NA
full <- rbind(train, test)

str(full)
dim(full)

lapply(full, function(x) length(unique(x))) 

summary(full)


###	Look at the first six rows or first few rows
head(full) 
head(full, 2) 



##Looking at missings 

sapply(full, function(x) {sum(is.na(x))})

full[1:10,]

##Removing missings


sapply(full, function(x) {sum(is.na(x))})

full[1:10,]


##Removing missings
sapply(full, function(x) {sum(is.na(x))})

fmedian <- median(full$Fare, na.rm = TRUE)
full[is.na(full$Fare), "Fare"] <- fmedian

full[is.na(full$Embarked), "Embarked"] <- "S"


full <- select(full, -Cabin, -Ticket)

outlierfilter <- full$Age < boxplot.stats(full$Age)$stats[5]


ageimputemodel <- lm(Age ~ Pclass + Sex + Fare + SibSp + Parch + Embarked, data = full[outlierfilter,])
newdata <- full[is.na(full$Age), c("Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked")]
full[is.na(full$Age), "Age"] <- predict(ageimputemodel, newdata)

sapply(full, function(x) {sum(is.na(x))})

full$agegrp[full$Age < 18] <- "Child"
full$agegrp[full$Age >= 18] <- "Adult"

# There are a lot of zeros in Parch but maybe ocombined with SibSP 
# let's look at the distribution

full$familysize <- full$SibSp + full$Parch + 1


full$faregrp[full$Fare<=7.896] <- "Low"
full$faregrp[full$Fare > 7.896 & full$Fare <= 14.454] <- "Mediuw low"
full$faregrp[full$Fare > 14.454 & full$Fare <= 31.275] <- "Medium"
full$faregrp[full$Fare > 31.275] <- "High"

count(full, 'faregrp')


## Let's do something about names
head(full$Name)

full$titles <- gsub("^.*, (.*?)\\..*$", "\\1", full$Name)
table(full$Sex, full$titles)

full$titles[full$titles %in% c("Mlle", "Ms")] <- "Miss" 
full$titles[full$titles == "Mme"] <- "Mrs"
full$titles[full$titles %in%  c("Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "the Countess")] <- "Other"
table(full$Sex, full$titles)

train <- full[full$set == "train", ]
test <- full[full$set == "test", ]



## Looking at relationships

##Tables
table(train[,c('Survived', 'Pclass')])

table(train[,c('Survived', 'Sex')])

table(train[,c('Survived', 'SibSp')])

## Box plots
bplot.xy(full$Survived, full$Age)

bplot.xy(full$Survived, full$Pclass)

bplot.xy(full$Survived, full$Fare)

bplot.xy(full$Survived, full$familysize)

bplot.xy(full$Survived, full$SibSp)


## GGplot


ggplot(full[!is.na(full$Survived),], aes(x = Survived, fill = Survived)) +
  geom_bar(stat='count') +
  geom_label(stat='count',aes(label=..count..), size=7) +
  theme_grey(base_size = 18)

ggplot(train, aes(x = Sex, fill = factor(Survived))) + 
  geom_bar(stat = 'count', position = "dodge") + 
  labs(x = "Sex", fill = "Survived", 
       title = "Survival by Sex") +
  theme(legend.position = c(0.9, 0.8), panel.background = NULL)

ggplot(train, aes(x = Pclass, fill = factor(Survived))) + 
  geom_bar(stat = 'count', position = "dodge") + 
  labs(x = "Pclass", fill = "Survived", 
       title = "Survivalship by Class") +
  theme(legend.position = c(0.9, 0.8), panel.background = NULL)

ggplot(train, aes(x = SibSp, fill = factor(Survived))) + 
  geom_bar(stat = 'count', position = "dodge") + 
  labs(x = "SibSp", fill = "Survived", 
       title = "Survivalship by Siblings") +
  theme(legend.position = c(0.9, 0.8), panel.background = NULL)

ggplot(train, aes(x = familysize, fill = factor(Survived))) + 
  geom_bar(stat = 'count', position = "dodge") + 
  labs(x = "family size", fill = "Survived", 
       title = "Survivalship by Siblings") +
  theme(legend.position = c(0.9, 0.8), panel.background = NULL)

ggplot(train, aes(x = agegrp, fill = factor(Survived))) + 
  geom_bar(stat = 'count', position = "dodge") + 
  labs(x = "Agegroup", fill = "Survived", 
       title = "Survivalship by Age") +
  theme(legend.position = c(0.9, 0.8), panel.background = NULL)

ggplot(train, aes(x = titles, fill = factor(Survived))) + 
  geom_bar(stat = 'count', position = "dodge") + 
  labs(x = "titles", fill = "Survived", 
       title = "Survivalship by titles") +
  theme(legend.position = c(0.9, 0.8), panel.background = NULL)

ggplot(train, aes(x = faregrp, fill = factor(Survived))) + 
  geom_bar(stat = 'count', position = "dodge") + 
  labs(x = "faregrp", fill = "Survived", 
       title = "Survivalship by faregrp") +
  theme(legend.position = c(0.9, 0.8), panel.background = NULL)


##Class is clearly important... Sex is importnat.

##Logistic


logmodel <- glm(Survived ~ Age + Pclass + Sex, family = "binomial", data = train)

summary(logmodel)

logmodel2 <- glm(Survived ~ Age + Pclass + Sex + SibSp , family = "binomial", data = train)

summary(logmodel2)

logmodel2 <- glm(Survived ~ Age + Pclass + Sex + familysize , family = "binomial", data = train)

summary(logmodel2)

logmodel3 <- glm(Survived ~ Age + Pclass + Sex + SibSp + Parch , family = "binomial", data = train)

summary(logmodel3)

logmodel4 <- glm(Survived ~ Age + Pclass + Sex + SibSp  + Fare, family = "binomial", data = train)

summary(logmodel4)

logmodel5 <- glm(Survived ~ Age + Pclass + Sex + SibSp  + titles, family = "binomial", data = train)

summary(logmodel5)

logmodel6 <- glm(Survived ~ Age + Pclass + Sex + familysize  + titles, family = "binomial", data = train)

summary(logmodel6)

logmodel7 <- glm(Survived ~ Age + Pclass + Sex + familysize  + titles, family = "binomial", data = train)

summary(logmodel7)

## Model 6 has lowest AIC, 


train$fitted <- round(fitted(logmodel6, train), digit=0)

cbind(train, fitted = fitted(logmodel6))

train$fitted <- round(fitted(logmodel6), digit=0)

train$match <- as.numeric(train$Survived == train$fitted)

count(train, 'match')

## 83 percent accuracy


predictTest = predict(logmodel6, type = "response", newdata = test)

# no preference over error t = 0.5
test$Survived = as.numeric(predictTest >= 0.5)

table(test$Survived)
head(test)

predictions = data.frame(test[c("PassengerId","Survived")])
write.csv(file = "TitanicPred_Logistic", x = predictions)


## Prediction with classification trees


surv_model <- rpart(formula = Survived ~ Age + Pclass + Sex + familysize  + titles + Embarked, data = train, method = "class")

# Print the results
print(surv_model)

head(train)

prediction <- predict(object = surv_model, newdata = train, type = "class")  

train$prediction <- prediction

train$match2 <- as.numeric(train$Survived == train$prediction)

count(train, 'match2')

## Second way to get same value
confusionMatrix(table(prediction, train$Survived))

fancyRpartPlot(surv_model)

##84 percent accuracy


## Making predictions

prediction <- predict(object = surv_model, newdata = test, type = "class")  

test$prediction <- prediction

head(test)

predictions2 = data.frame(test[c("PassengerId","prediction")])
write.csv(file = "TitanicPred_DecisionTree", x = predictions2)



## Predictions with Random Forest

head(full)

full$Pclass <- as.factor(full$Pclass)
full$Sex <- as.factor(full$Sex)
full$titles <- as.factor(full$titles)


train <- full[full$set == "train", ]
test <- full[full$set == "test", ]

set.seed(111)

forestmodel <- randomForest(as.factor(Survived) ~ Age + Pclass + Sex + familysize  + titles + Fare, data=train, importance=T, ntree=1000, metric = "Accuracy")

forestmodel


# 84 percent accuracy 

plot(forestmodel, ylim=c(0,0.36))
legend('topright', colnames(forestmodel$err.rate), col=1:3, fill=1:3)


my_prediction <- predict(forestmodel, test)

test$my_prediction <- my_prediction

predictions3 = data.frame(test[c("PassengerId","my_prediction")])
write.csv(file = "TitanicPred_RandomForest", x = predictions3)

varImpPlot(forestmodel)

head(test)



