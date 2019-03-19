#packages
require(pROC)
require(caret)
require(woe)
require(randomForest)

#data import
setwd("C:/Users/Arvind B S/Documents/BAPM Course Material/1st Sem/Predictive Modeling/Group Project/")
data<-read.csv("data_clean.csv")
str(data)

#feature engineering
cat_vars<-Filter(is.factor, data)
dummies<-dummyVars(~.,data = cat_vars)
dum<-predict(dummies,data)
str(data.frame(dum))
write.csv(dum,file = "ind_vars.csv")

woe(Data = data,Independent = "age",Continuous = TRUE,Dependent = "no_show",C_Bin = 4,Bad = 0,Good = 1)
woe(Data = data,Independent = "n_visit",Continuous = TRUE,Dependent = "no_show",C_Bin = 5,Bad = 0,Good = 1)
woe(Data = data,Independent = "day_gap_floor",Continuous = TRUE,Dependent = "no_show",C_Bin = 5,Bad = 0,Good = 1)

#modeling
data_model<-read.csv("data_clean.csv")
cols<-seq(3,length(data_model))
data_model[cols]<-lapply(data_model[cols],factor)
str(data_model)


#Resample
ones<-data_model[which(data_model$no_show=='1'),]
zeros<-data_model[which(data_model$no_show=='0'),]
nrow(ones)
nrow(zeros)
resmp_size <-floor(0.5 * nrow(ones))
#set.seed(123)
resmp_train_ones_ind <- sample(seq_len(nrow(ones)), size = resmp_size)
resmp_train_zeros_ind <- sample(seq_len(nrow(zeros)), size = resmp_size)

resmp_train <- rbind(ones[resmp_train_ones_ind,],zeros[resmp_train_zeros_ind,])                                
resmp_test<- rbind(ones[-resmp_train_ones_ind,],zeros[-resmp_train_zeros_ind,])

#logistic on train
logistic<-glm(no_show~.,data = resmp_train,family = binomial(link = logit))
summary(logistic)
#train roc
prob_logistic<-predict(logistic,type=c("response"),newdata = resmp_train)
roc_curve(prob_logistic,resmp_train$no_show,resmp_train)
#test roc
prob_logistic<-predict(logistic,newdata = resmp_test)
roc_curve(prob_logistic,resmp_test$no_show,resmp_test)







## train-test split 75-25
smp_size <-floor(0.75 * nrow(data_model))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_model)), size = smp_size)
train <- data_model[train_ind, ]
test <- data_model[-train_ind, ]


roc_curve<-function(prob,dv,data)
{
  data$prob<-prob
  g<-roc(dv~prob,data = data,cutoff = c(0.7,0.3))
  print(auc(g))
  plot(g)
}


#logistic on train
logistic<-glm(no_show~.,data = train,family = binomial(link = logit))
summary(logistic)
#train roc
prob_logistic<-predict(logistic,type=c("response"),newdata = train)
roc_curve(prob_logistic,train$no_show,train)
#test roc
prob_logistic_test<-predict(logistic,newdata = test)
roc_curve(prob_logistic_test,test$no_show,test)


confusionMatrix(data = as.numeric(prob_logistic>0.5), reference = test$no_show)
confusionMatrix(data = as.numeric(prob_logistic>0.3), reference = test$no_show)
confusionMatrix(data = as.numeric(prob_logistic>0.2), reference = test$no_show)


#RF on train
rf<-randomForest(no_show~.,data = train,ntree = 500,replace = TRUE,cutoff=c(0.7,0.3))
rf
importance(rf)
#train roc
prob_rf<-predict(rf,newdata = train,type = "prob")
roc_curve(prob_rf,train$no_show,train)

#test roc
prob_rf<-predict(rf,newdata = test)
roc_curve(prob_rf,test$no_show,test)



#rf
fitControl <- trainControl(method = "cv", number = 7)
set.seed(123)
rfFit1 <- train(as.factor(no_show)  ~ ., 
                 data = train, method = "rf", trControl = fitControl,verbose = FALSE)
train$prob_rf<-predict(rfFit1, train,type= "prob")[,2]
test$prob_rf<-predict(rfFit1, test,type= "prob")[,2]
#train auc
auc(train$no_show,train$prob_rf)
#test auc
auc(test$no_show,test$prob_rf)
#confusion matrix
train_pred<-predict(rfFit1, train)
confusionMatrix(data = train_pred,train$no_show)
test_pred<-predict(rfFit1, test)
confusionMatrix(data = test_pred,test$no_show)


#gbm
fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)
set.seed(123)
gbmFit1 <- train(as.factor(no_show)  ~ ., 
                 data = train, method = "gbm", trControl = fitControl,verbose = FALSE)
train$prob_gbm<-predict(gbmFit1, train,type= "prob")[,2]
test$prob_gbm<-predict(gbmFit1, test,type= "prob")[,2]
#train auc
auc(train$no_show,train$prob_gbm)
#test auc
auc(test$no_show,test$prob_gbm)
#confusion matrix
train_pred<-predict(gbmFit1, train)
confusionMatrix(data = train_pred,train$no_show)
test_pred<-predict(gbmFit1, test)
confusionMatrix(data = test_pred,test$no_show)


#nnet
fitControl <- trainControl(method = "cv", number = 4)
set.seed(123)
nnetFit1 <- train(as.factor(no_show)  ~ ., 
                 data = train, method = "nnet", trControl = fitControl,verbose = FALSE)
train$prob_nnet<-predict(nnetFit1, train,type= "prob")[,2]
test$prob_nnet<-predict(nnetFit1, test,type= "prob")[,2]
#train auc
auc(train$no_show,train$prob_nnet)
#test auc
auc(test$no_show,test$prob_nnet)
#confusion matrix
train_pred<-predict(nnetFit1, newdata = train)
test_pred<-predict(nnetFit1, newdata = test)
confusionMatrix(data = test_pred, reference = test$no_show)

#Ensemble
prob_ens_max<-max(prob_logistic_test,prob_gbm_test,prob_nnet_test)
prob_ens_mean<-mean(prob_logistic_test,prob_gbm_test,prob_nnet_test)
prob_ens_median<-median(prob_logistic_test,prob_gbm_test,prob_nnet_test)


set.seed(123)
cv_splits<-createFolds(data_model$no_show,k = 10)
str(cv_splits)
cv_splits[1]
cv_splits$Fold01

for (fold in cv_splits)
{
  cv_train<-data_model[-fold,]
  cv_test<-data_model[fold,]
  glm(no_show~.,data = cv_train)
  
}


#Learning Curve
#install.packages("InformationValue")
require(InformationValue)
## train-test split 75-25
train_misclass<-c(0)
test_misclass<-c(0)
for (i in seq(0.3,0.9,0.1))
{
  smp_size <-floor(i * nrow(data_model))
  set.seed(123)
  train_ind <- sample(seq_len(nrow(data_model)), size = smp_size)
  train <- data_model[train_ind, ]
  test <- data_model[-train_ind, ]
  logistic<-glm(no_show~.,data = train,family = binomial(link = logit))
  prob_train<-predict(logistic,type=c("response"),newdata = train)
  prob_test<-predict(logistic,type=c("response"),newdata = test)
  train_error<-misClassError(train$no_show,prob_train,threshold = 0.5)
  test_error<-misClassError(test$no_show,prob_test,threshold = 0.5)
  train_misclass<-c(train_misclass,train_error)
  test_misclass<-c(test_misclass,test_error)  
};

smp_size <-floor(0.3 * nrow(data_model))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_model)), size = smp_size)
train <- data_model[train_ind, ]
test <- data_model[-train_ind, ]
logistic<-glm(no_show~.,data = train,family = binomial(link = logit))
prob_train<-predict(logistic,type=c("response"),newdata = train)
prob_test<-predict(logistic,type=c("response"),newdata = test)
train_error<-misClassError(train$no_show,prob_train,threshold = 0.5)
test_error<-misClassError(test$no_show,prob_test,threshold = 0.5)
train_misclass<-c(train_misclass,train_error)
test_misclass<-c(test_misclass,test_error)  

smp_size <-floor(0.4 * nrow(data_model))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_model)), size = smp_size)
train <- data_model[train_ind, ]
test <- data_model[-train_ind, ]
logistic<-glm(no_show~.,data = train,family = binomial(link = logit))
prob_train<-predict(logistic,type=c("response"),newdata = train)
prob_test<-predict(logistic,type=c("response"),newdata = test)
train_error<-misClassError(train$no_show,prob_train,threshold = 0.5)
test_error<-misClassError(test$no_show,prob_test,threshold = 0.5)
train_misclass<-c(train_misclass,train_error)
test_misclass<-c(test_misclass,test_error)  

smp_size <-floor(0.5 * nrow(data_model))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_model)), size = smp_size)
train <- data_model[train_ind, ]
test <- data_model[-train_ind, ]
logistic<-glm(no_show~.,data = train,family = binomial(link = logit))
prob_train<-predict(logistic,type=c("response"),newdata = train)
prob_test<-predict(logistic,type=c("response"),newdata = test)
train_error<-misClassError(train$no_show,prob_train,threshold = 0.5)
test_error<-misClassError(test$no_show,prob_test,threshold = 0.5)
train_misclass<-c(train_misclass,train_error)
test_misclass<-c(test_misclass,test_error)  

smp_size <-floor(0.6 * nrow(data_model))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_model)), size = smp_size)
train <- data_model[train_ind, ]
test <- data_model[-train_ind, ]
logistic<-glm(no_show~.,data = train,family = binomial(link = logit))
prob_train<-predict(logistic,type=c("response"),newdata = train)
prob_test<-predict(logistic,type=c("response"),newdata = test)
train_error<-misClassError(train$no_show,prob_train,threshold = 0.5)
test_error<-misClassError(test$no_show,prob_test,threshold = 0.5)
train_misclass<-c(train_misclass,train_error)
test_misclass<-c(test_misclass,test_error)  

smp_size <-floor(0.7 * nrow(data_model))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_model)), size = smp_size)
train <- data_model[train_ind, ]
test <- data_model[-train_ind, ]
logistic<-glm(no_show~.,data = train,family = binomial(link = logit))
prob_train<-predict(logistic,type=c("response"),newdata = train)
prob_test<-predict(logistic,type=c("response"),newdata = test)
train_error<-misClassError(train$no_show,prob_train,threshold = 0.5)
test_error<-misClassError(test$no_show,prob_test,threshold = 0.5)
train_misclass<-c(train_misclass,train_error)
test_misclass<-c(test_misclass,test_error)  

smp_size <-floor(0.8 * nrow(data_model))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_model)), size = smp_size)
train <- data_model[train_ind, ]
test <- data_model[-train_ind, ]
logistic<-glm(no_show~.,data = train,family = binomial(link = logit))
prob_train<-predict(logistic,type=c("response"),newdata = train)
prob_test<-predict(logistic,type=c("response"),newdata = test)
train_error<-misClassError(train$no_show,prob_train,threshold = 0.5)
test_error<-misClassError(test$no_show,prob_test,threshold = 0.5)
train_misclass<-c(train_misclass,train_error)
test_misclass<-c(test_misclass,test_error)  

smp_size <-floor(0.9 * nrow(data_model))
set.seed(123)
train_ind <- sample(seq_len(nrow(data_model)), size = smp_size)
train <- data_model[train_ind, ]
test <- data_model[-train_ind, ]
logistic<-glm(no_show~.,data = train,family = binomial(link = logit))
prob_train<-predict(logistic,type=c("response"),newdata = train)
prob_test<-predict(logistic,type=c("response"),newdata = test)
train_error<-misClassError(train$no_show,prob_train,threshold = 0.5)
test_error<-misClassError(test$no_show,prob_test,threshold = 0.5)
train_misclass<-c(train_misclass,train_error)
test_misclass<-c(test_misclass,test_error)  

 

size<-seq(0.3,0.9,0.1)
size
train_misclass<-train_misclass[c(seq(2,8))]
test_misclass<-test_misclass[c(seq(2,8))]
plot(size,train_misclass)
plot(size,test_misclass)

