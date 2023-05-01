library(randomForest)   
library(caret)
library(dplyr)
library(magrittr)

###############################
########## Codes from lab6
###############################
###### read and preprocess the data
df <- load('df_train_test.RData')
df_train %<>% cbind(as.factor(population_train)) %>% rename(labels = "as.factor(population_train)")
df_test %<>% cbind(as.factor(population_test)) %>% rename(labels = 'as.factor(population_test)')

############################
########## RF
############################
set.seed(123)
### setting 10-fold CV
control <- trainControl(method='cv', 
                        number=10, 
                        search = 'grid',
                        verboseIter = TRUE)

## specify the grid of mtry to test
tunegrid<-expand.grid(.mtry = 1:(ncol(df_train)-1))

############################
## also try to tune "ntree"
############################
modellist <- list()
#train with different ntree parameters
for (ntree in c(500,1000,1500,2000)){
  set.seed(123)
  fit <- train(labels ~ .,
               data = df_train,
               method = 'rf',
               metric = 'Accuracy',
               tuneGrid  = tunegrid, 
               trControl = control,
               ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}

####### Save the CV results
#saveRDS(modellist,'model_list.rds')
####### Load the CV results
modellist <- readRDS('model_list.rds')

####### Check the tuning parameter, ntree=1000 has the highest accuracy
results <- resamples(modellist)
summary(results)


###########################
####### Final result
###########################

## Extract the final model
## ntree=1000, mtry=42
final_model <- modellist[['1000']]$finalModel


#### Accuracy of test data = 0.7549669
df_pred <- predict(final_model, newdata = df_test)
mean(df_pred == df_test$labels)

##### Confusion Matrix of test data
# Reference
# Prediction CEU FIN GBR IBS TSI
# CEU  14   0  10   1   0
# FIN   0  34   0   1   0
# GBR  10   2  15   1   1
# IBS   2   0   2  25   5
# TSI   1   0   1   0  26
confusionMatrix(df_pred, as.factor(df_test$labels))
