rm(list = ls())

# Set timer for script run
st = list()
st[['Start']] = Sys.time()

options(max.print = 200000)

###############################################################################################################
###############################################################################################################

## load libraries
library(caret)
library(dplyr)
library(usdm)
library(MLmetrics)
library(skimr)
library(ggplot2)
library(jpeg)
library(ROSE)
library(randomForest)
library(e1071)

###############################################################################################################
###############################################################################################################

## setting the working directory
setwd('/Users/darsh/Documents/Data Science/Santander Project/R')

###############################################################################################################
###############################################################################################################

## load dataset
df.train = read.csv('train.csv')
df.test = read.csv('test.csv')

# Backing up the ID_code field
key.train = df.train$ID_code
key.test = df.test$ID_code

# Removing ID variables
df.train$ID_code = NULL
df.test$ID_code = NULL

###############################################################################################################
###############################################################################################################

# Check for missing values
anyNA(df.train)

################################## No missing values found ####################################################

## Preparing data for analysis
training = df.train

# Checking multicollinearity
vifcor(training)

#Bringing all features to a single scale
training[,2:201] = scale(training[,2:201])
df.test[,1:200] = scale(training[,1:200])

# Reformatting the levels of the target variable class
training$target = as.factor(training$target)
levels(training$target) = c('zero', 'one')

# Splitting dataframe into independent and dependent variables

independent_var = colnames(training) != 'target'
x = training[,independent_var]
y = training$target

#################### No variable from the 200 input variables has collinearity problem. #######################

# Statistical summary with histograms
skimmed = skim_to_wide(x)
print(skimmed)

# Create a generic function for generating distribution & box plots
buildplot =function(x, type = 'dist'){
  variablenames = colnames(x)
  temp = as.numeric(1)
  for(i in seq(8, dim(x)[2], 8)){
    par(mar = c(2, 2, 2, 2))
    par(mfrow = c(4, 2))
    for (j in variablenames[temp:i]){
      if(type == 'box'){
        jpeg(filename = paste('boxplot', j, '.jpg', sep = ' '), width = 1080, height = 1080)
        boxplot(x[[j]], main = j, col = 'grey' )
        dev.off()
      }
        
      else{
        jpeg(filename = paste('dplot', j, '.jpg', sep = ' '), width = 1080, height = 1080)
        plot(density(x[[j]]), main = j, col = 'red' )
        dev.off()
      }
    }
    temp = i + 1
  }
}


# Distribution plot for training & test data 
buildplot(x)
buildplot(df.test)

######### The distribution of the independant variables in both train & test datasets is fairly normal ########

# Box plot for outlier analysis of training & test data 
buildplot(x, 'box')
buildplot(df.test, 'box')

######### The distribution of the independant variables in both train & test datasets is fairly normal ########

###############################################################################################################
###############################################################################################################

## Partitioning the train dataset for validation

# Backing up the complete processed train dataset
training_complete = training

# create a list of 75% of the rows in the train dataset we can use for training
validation_index = createDataPartition(training$target, p=0.75, list=FALSE)

#select 20% of the data for validation
validation = training[-validation_index,]

# use the remaining 80% of data to training and testing the models
training = training[validation_index,]

# Creating a stratified sample with oversampled positive target response
training_over = ovun.sample(target ~ ., training, method = 'over')$data

###############################################################################################################
###############################################################################################################

## Evaluating algorithms

# Build models
set.seed(2)
fit.glm = glm(target ~., data = training, family = binomial)

set.seed(2)
fit.rf = randomForest(target ~., data = training, ntree = 20, mtry = 10)

set.seed(2)
fit.naivebayes = naiveBayes(target ~., data = training)

###############################################################################################################
###############################################################################################################

#test GLM on validation subset
# print(fit.glm)
predictions.glm = predict(fit.glm, validation)
predictions.glm = as.factor(ifelse(predictions.glm < 0.5, 'zero', 'one'))
cm = ConfusionMatrix(predictions.glm, validation$target)
glm = list()
glm[['accuracy']] = Accuracy(predictions.glm, validation$target) * 100
glm[['fnr']] = (cm['one','zero'] * 100)/(cm['one','zero']+cm['one', 'one'])
glm[['auc']] = roc.curve(predictions.glm, validation$target)$auc
glm[['ConfusionMatrix']] = cm

#test RandomForest on validation subset
# print(fit.xgb)
predictions.rf = predict(fit.rf, validation)
cm = ConfusionMatrix(predictions.rf, validation$target)
rf = list()
rf[['accuracy']] = Accuracy(predictions.rf, validation$target) * 100
rf[['auc']] = roc.curve(predictions.rf, validation$target)$auc
rf[['fnr']] = (cm['one','zero'] * 100)/(cm['one','zero']+cm['one', 'one'])
rf[['ConfusionMatrix']] = cm

#test NaiveBayes on validation subset
# print(fit.naivebayes)
predictions.naivebayes = predict(fit.naivebayes, validation)
cm = ConfusionMatrix(predictions.naivebayes, validation$target)
naivebayes = list()
naivebayes[['accuracy']] = Accuracy(predictions.naivebayes, validation$target) * 100
naivebayes[['auc']] = roc.curve(predictions.naivebayes, validation$target)$auc
naivebayes[['fnr']] = (cm['one','zero'] * 100)/(cm['one','zero']+cm['one', 'one'])
naivebayes[['ConfusionMatrix']] = cm

results = list('glm' = glm, 'rf' = rf, 'naivebayes' = naivebayes)

###############################################################################################################
###############################################################################################################

## Evaluating algorithms with oversampled training dataset

# Build models
set.seed(2)
over.glm = glm(target ~., data = training_over, family = binomial)

set.seed(2)
over.rf = randomForest(target ~., data = training_over, ntree = 20, mtry = 10)

set.seed(2)
over.naivebayes = naiveBayes(target ~., data = training_over)

###############################################################################################################
###############################################################################################################

#test GLM on validation subset
# print(fit.glm)
predictions.glm = predict(over.glm, validation)
predictions.glm = as.factor(ifelse(predictions.glm < 0.5, 'zero', 'one'))
cm = ConfusionMatrix(predictions.glm, validation$target)
glm = list()
glm[['accuracy']] = Accuracy(predictions.glm, validation$target) * 100
glm[['auc']] = roc.curve(predictions.glm, validation$target)$auc
glm[['fnr']] = (cm['one','zero'] * 100)/(cm['one','zero']+cm['one', 'one'])
glm[['ConfusionMatrix']] = cm

#test RandomForest on validation subset
# print(fit.xgb)
predictions.rf = predict(over.rf, validation)
cm = ConfusionMatrix(predictions.rf, validation$target)
rf = list()
rf[['accuracy']] = Accuracy(predictions.rf, validation$target) * 100
rf[['auc']] = roc.curve(predictions.rf, validation$target)$auc
rf[['fnr']] = (cm['one','zero'] * 100)/(cm['one','zero']+cm['one', 'one'])
rf[['ConfusionMatrix']] = cm

#test NaiveBayes on validation subset
# print(fit.naivebayes)
predictions.naivebayes = predict(over.naivebayes, validation)
cm = ConfusionMatrix(predictions.naivebayes, validation$target)
naivebayes = list()
naivebayes[['accuracy']] = Accuracy(predictions.naivebayes, validation$target) * 100
naivebayes[['auc']] = roc.curve(predictions.naivebayes, validation$target)$auc
naivebayes[['fnr']] = (cm['one','zero'] * 100)/(cm['one','zero']+cm['one', 'one'])
naivebayes[['ConfusionMatrix']] = cm

results_over = list('glm' = glm, 'rf' = rf, 'naivebayes' = naivebayes)

###############################################################################################################
###############################################################################################################

# Selecting NaiveBayes algorithm modeled using complete train data to predict the test data
# since it provides the best ratio of Accuracy and FNR

# Creating a stratified sample with oversampled positive target response
training_over = ovun.sample(target ~ ., training_complete, method = 'over')$data

set.seed(2)
over.naivebayes = naiveBayes(target ~., data = training_over)

#test NaiveBayes on validation subset
# print(fit.naivebayes)
predictions.naivebayes = predict(over.naivebayes, df.test)
df.test$target = predictions.naivebayes
levels(df.test$target) = c('0','1')

# Saving the final dataset
write.csv(df.test, 'submission - r.csv')

st[['End']] = Sys.time()

print(paste('Total Script Runtime: ', st$End - st$Start, sep = ''))