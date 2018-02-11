
# Load caret libraries
library(caret)
library(caretEnsemble)

# Load the dataset
hr_data <-  read.csv("/Users/sibanjan/datascience/dzone/caret/data/hr.csv")
hr_data$left <- as.factor(hr_data$left)
levels(hr_data$left) <- c("stayed", "left")

# Create train and test data sets
trainIndex = createDataPartition(hr_data$left, p=0.7, list=FALSE,times=1)
 
train_set = hr_data[trainIndex,]
test_set = hr_data[-trainIndex,]

seed <- 999
metric <- "Accuracy"

# Bagging Algorithm (Random Forest)
# Parameters used to control the model training process are defined in trainControl method
bagcontrol <- trainControl(sampling="rose",method="repeatedcv", number=5, repeats=3)
set.seed(seed)
#"rf" method is for training random forest  model
fit.rf <- train(left~., data=train_set, method="rf", metric=metric, trControl=bagcontrol)
# evaluate results on test set
test_set$pred <- predict(fit.rf, newdata=test_set)
confusionMatrix(data = test_set$pred, reference = test_set$left)

# Gradient Boosting

boostcontrol <- trainControl(sampling="rose",method="repeatedcv", number=5, repeats=2)
set.seed(seed)
fit.gbm <- train(left~., data=train_set, method="gbm", metric=metric, trControl=boostcontrol, verbose=FALSE)
# evaluate results on test set
test_set$pred <- predict(fit.gbm, newdata=test_set)
confusionMatrix(data = test_set$pred, reference = test_set$left)



# Stacking Algorithms

control <- trainControl(sampling="rose",method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c( 'knn','glm','rpart')
set.seed(seed)
stack_models <- caretList(left~., data=train_set, trControl=control, methodList=algorithmList)
stacking_results <- resamples(stack_models)
summary(stacking_results)
dotplot(stacking_results)

# Check correlation between models to ensure the results are uncorrelated and can be ensembled
modelCor(stacking_results)
splom(stacking_results)

# stack using Logistics Regression
stackControl <- trainControl(sampling="rose",method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(stack_models, method="glm", metric=metric, trControl=stackControl)
print(stack.glm)
# evaluate results on test set
test_set$pred <- predict(stack.glm, newdata=test_set)
confusionMatrix(data = test_set$pred, reference = test_set$left)

# stack using gbm
set.seed(seed)
stack.gbm <- caretStack(stack_models, method="gbm", metric=metric, trControl=stackControl)
print(stack.gbm)
test_set$pred <- predict(stack.gbm, newdata=test_set)
confusionMatrix(data = test_set$pred, reference = test_set$left)