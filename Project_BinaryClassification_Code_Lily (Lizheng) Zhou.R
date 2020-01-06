rm(list = ls())    # delete objects
cat("\014")
library(class)
library(ggplot2)
library(dplyr)
library(glmnet)
library(MASS)
library(rmutil)
library(tictoc)
library(latex2exp)
library(randomForest)
library(e1071)
library(gridExtra)
library(tidyverse)
library(RColorBrewer)
library(coefplot)
#library(filter)

# import full datset (n=10000, p=88)
data10000	    =	   read.csv("D:/d/Courses/STA/STA 9891/Project/hedge-fund-x-financial-modeling-challenge/deepanalytics_dataset.csv",header=TRUE)
y             =    factor(data10000$target)
# test data is imbalance or not
sum(y==0)
sum(y==1)

# decided to use a subset of full dataset (n=1000, p=88), 
# because time complexity of SVM is n2, which is too time consuming. 
data1000	    =	   read.csv("D:/d/Courses/STA/STA 9891/Project/hedge-fund-x-financial-modeling-challenge/deepanalytics_dataset_1000.csv",header=TRUE)
X             =    model.matrix(target~., data1000)[, -1]
y             =    factor(data1000$target)
# test subset of data is imbalance or not
sum(y==0)
sum(y==1)

# preparation
n             =    dim(X)[1] # sample size
p             =    dim(X)[2] # number of predictors/features
S             =    100

learn.pct1    =    0.5
learn.pct2    =    0.9
learn.pct     =    c(learn.pct1, learn.pct2)
random_order  =    matrix(0, nrow = S, ncol = n) 

set.seed(1) 

X             =    scale(X)

# Err is S x 14 matrix
# column  1 of Err = total train error (n learn = 0.5n)
# column  2 of Err = total train error (n learn = 0.9n)

# column  3 of Err = total test error (n learn = 0.5n)
# column  4 of Err = total test error (n learn = 0.9n)

# column  5 of Err = min cv error (n learn = 0.5n)
# column  6 of Err = min cv error (n learn = 0.9n)

# column  7 of Err = false positive train error (n learn = 0.5n)
# column  8 of Err = false negative train error (n learn = 0.5n)
# column  9 of Err = false positive test error (n learn = 0.5n)
# column 10 of Err = false negative test error (n learn = 0.5n)

# column 11 of Err = false positive train error (n learn = 0.9n)
# column 12 of Err = false negative train error (n learn = 0.9n)
# column 13 of Err = false positive test error (n learn = 0.9n)
# column 14 of Err = false negative test error (n learn = 0.9n)

# set up error matrix for each methods
Err.rf        =    matrix(0, nrow = S, ncol = 14) 
Err.svm       =    matrix(0, nrow = S, ncol = 14) 
Err.logistic  =    matrix(0, nrow = S, ncol = 14) 
Err.lasso     =    matrix(0, nrow = S, ncol = 14) 
Err.ridge     =    matrix(0, nrow = S, ncol = 14) 

# fit models 100 times with 5 methods
for (s in 1:S) {
  
  # randomly splitting the data into test and train set
  random_order[s,]  =  sample(n)
  
  for (i in learn.pct) {
    
    # record time
    ptm                 =     proc.time()
    
    n.train      =  floor(n*i)
    n.test       =  n-n.train
    trainSet     =  random_order[s,][1:n.train]
    testSet      =  random_order[s,][(1+n.train):n] 
    X.train      =  X[trainSet, ]
    y.train      =  y[trainSet]
    X.test       =  X[testSet, ]
    y.test       =  y[testSet]
    y.os.train    =  y.train   # initialize the over-sampled (os) set to train the models
    X.os.train    =  X.train   # initialize the over-sampled (os) set to train the models
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # to take into account the imbalance 
    # below we over-sample (with replacement) the  data so that the data is balanced
    imbalance     =     FALSE   
    if (imbalance == TRUE) {
      index.yis0      =      which(y.train==0)  # idetify the index of  points with label 0
      index.yis1      =      which(y.train==1) # idetify the index of  points with label 1
      n.train.1       =      length(index.yis1)
      n.train.0       =      length(index.yis0)
      if (n.train.1 > n.train.0) {     # we need more 0s in out training set, so we over sample with replacement
        more.train    =      sample(index.yis0, size=n.train.1-n.train.0, replace=TRUE)
      }         else {    # we need more 1s in out training set, so we over sample with replacement          
        more.train    =      sample(index.yis1, size=n.train.0-n.train.1, replace=TRUE)
      }
      ##### the code below CORRECTLY over samples the train set 
      ##### and stores it in y.train_ and X.train_
      y.os.train        =       as.factor(c(y.train, y.train[more.train])-1) 
      X.os.train        =       rbind2(X.train, X.train[more.train,])    
      
      ##### the code below is MISTAKENLY over sample the train set  
      # trainSet       =       rbind(trainSet, more.train)
      # y.train        =       y[trainSet]
      # X.train        =       X[trainSet, ]
      # n.train        =       dim(X.train)[1] 
      ##################################################################################
    } # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # alternative way of breaking data into train and test 
    #    train.data              =   data.frame(X[trainSet, ], as.factor(y[trainSet]))
    #    test.data               =   data.frame(X[testSet, ], as.factor(y[testSet]))
    #    names(train.data)[89]   =   "target"
    #    names(test.data)[89]    =   "target"
    
    os.train.data           =   data.frame(X.os.train, as.factor(y.os.train))
    train.data              =   data.frame(X.train, as.factor(y.train))
    test.data               =   data.frame(X.test, as.factor(y.test))
    names(os.train.data)[89]=   "target"
    names(train.data)[89]   =   "target"
    names(test.data)[89]    =   "target"
    
    # random forrest
    rf.fit         =   randomForest(target~., data=os.train.data, mtry=sqrt(p), importance=TRUE)
    y.train.hat    =   predict(rf.fit, newdata=train.data)
    y.test.hat     =   predict(rf.fit, newdata=test.data)
    
    if (i == learn.pct1) {
      Err.rf[s,1]  =   mean(y.train != y.train.hat)
      Err.rf[s,3]  =   mean(y.test != y.test.hat)
      
      Err.rf[s,7]  =   mean(1 == y.train.hat[y.train==0]) # false positive
      Err.rf[s,8]  =   mean(0 == y.train.hat[y.train==1]) # false negative
      Err.rf[s,9]  =   mean(1 == y.test.hat[y.test==0]) # false positive
      Err.rf[s,10] =   mean(0 == y.test.hat[y.test==1]) # false negative
    } else {
      Err.rf[s,2]  =   mean(y.train != y.train.hat)
      Err.rf[s,4]  =   mean(y.test != y.test.hat)
      
      Err.rf[s,11] =   mean(1 == y.train.hat[y.train==0]) # false positive
      Err.rf[s,12] =   mean(0 == y.train.hat[y.train==1]) # false negative
      Err.rf[s,13] =   mean(1 == y.test.hat[y.test==0]) # false positive
      Err.rf[s,14] =   mean(0 == y.test.hat[y.test==1]) # false negative
    }
    
    # radial svm
    tune.svm    =   tune(svm, target ~ ., data=os.train.data, kernel = "radial",
                         ranges = list(cost = 10^seq(-2,2,length.out = 5),
                                       gamma = 10^seq(-2,2,length.out = 5) ))
    svm.fit     =   tune.svm$best.model
    
    y.train.hat =   predict(svm.fit, newdata=train.data)
    y.test.hat  =   predict(svm.fit, newdata=test.data)
    
    if (i == learn.pct1) {
      Err.svm[s,1]    =     mean(y.train != y.train.hat)
      Err.svm[s,3]    =     mean(y.test != y.test.hat)
      Err.svm[s,5]    =     tune.svm$best.performance
      
      Err.svm[s,7]    =     mean(1 == y.train.hat[y.train==0]) # false positive
      Err.svm[s,8]    =     mean(0 == y.train.hat[y.train==1]) # false negative
      Err.svm[s,9]    =     mean(1 == y.test.hat[y.test==0]) # false positive
      Err.svm[s,10]   =     mean(0 == y.test.hat[y.test==1]) # false negative
    } else {
      Err.svm[s,2]    =     mean(y.train != y.train.hat)
      Err.svm[s,4]    =     mean(y.test != y.test.hat)
      Err.svm[s,6]    =     tune.svm$best.performance
      
      Err.svm[s,11]   =     mean(1 == y.train.hat[y.train==0]) # false positive
      Err.svm[s,12]   =     mean(0 == y.train.hat[y.train==1]) # false negative
      Err.svm[s,13]   =     mean(1 == y.test.hat[y.test==0]) # false positive
      Err.svm[s,14]   =     mean(0 == y.test.hat[y.test==1]) # false negative
    }
    
    # logistic
    logistic.fit          =   glm(target ~ ., os.train.data, family="binomial")
    logistic.probs.train  =   predict(logistic.fit, train.data, "response")
    logistic.probs.test   =   predict(logistic.fit, test.data, "response")
    y.train.hat           =   rep(0, length(logistic.probs.train))
    y.test.hat            =   rep(0, length(logistic.probs.test))
    y.train.hat[logistic.probs.train > 0.5] = 1
    y.test.hat[logistic.probs.test > 0.5]   = 1
    
    #y.train.hat       =     predict(ridge.fit, newx = X.train, type = "class")
    #y.test.hat        =     predict(ridge.fit, newx = X.test, type = "class")
    
    if (i == learn.pct1) {
      Err.logistic[s,1]    =     mean(y.train != y.train.hat)
      Err.logistic[s,3]    =     mean(y.test != y.test.hat)
      
      Err.logistic[s,7]    =     mean(1 == y.train.hat[y.train==0]) # false positive
      Err.logistic[s,8]    =     mean(0 == y.train.hat[y.train==1]) # false negative
      Err.logistic[s,9]    =     mean(1 == y.test.hat[y.test==0]) # false positive
      Err.logistic[s,10]   =     mean(0 == y.test.hat[y.test==1]) # false negative
    } else {
      Err.logistic[s,2]    =     mean(y.train != y.train.hat)
      Err.logistic[s,4]    =     mean(y.test != y.test.hat)
      
      Err.logistic[s,11]   =     mean(1 == y.train.hat[y.train==0]) # false positive
      Err.logistic[s,12]   =     mean(0 == y.train.hat[y.train==1]) # false negative
      Err.logistic[s,13]   =     mean(1 == y.test.hat[y.test==0]) # false positive
      Err.logistic[s,14]   =     mean(0 == y.test.hat[y.test==1]) # false negative
    }
    
    # optimize lasso logistic regression using cross validation
    m                 =     25
    lasso.cv          =     cv.glmnet(X.os.train, y.os.train, family = "binomial", alpha = 1, nfolds = 10, type.measure="class")
    lasso.fit         =     glmnet(X.os.train, y.os.train, lambda = lasso.cv$lambda.min, family = "binomial", alpha = 1)
    
#    m             =    25
#    lasso.cv      =    cv.glmnet(X.os.train, y.os.train, family="binomial", alpha=1, 
#                                 nfolds=10, type.measure="class")
#    lam.lasso     =    exp(seq(log(max(lasso.cv$lambda)),log(0.00001), 
#                               (log(0.00001) - log(max(lasso.cv$lambda)))/(m-1)))
#    lasso.cv      =    cv.glmnet(X.os.train, y.os.train, lambda=lam.lasso, 
#                                 family="binomial", alpha=1,
#                                 nfolds=10, type.measure="class")
#    lasso.fit     =    glmnet(X.os.train, y.os.train, lambda=lasso.cv$lambda.min, 
#                              family="binomial", alpha=1)
    
#    m             =    25
#    lasso.cv      =    cv.glmnet(X.os.train, y.os.train, family="binomial", alpha=1, 
#                                 intercept=TRUE, standardize=FALSE, 
#                                 nfolds=10, type.measure="class")
#    lam.lasso     =    exp(seq(log(max(lasso.cv$lambda)),log(0.00001), 
#                               (log(0.00001) - log(max(lasso.cv$lambda)))/(m-1)))
#    lasso.cv      =    cv.glmnet(X.os.train, y.os.train, lambda=lam.lasso, 
#                                 family="binomial", alpha=1,
#                                 intercept=TRUE, standardize=FALSE, 
#                                 nfolds=10, type.measure="class")
#    lasso.fit     =    glmnet(X.os.train, y.os.train, lambda=lasso.cv$lambda.min, 
#                              family="binomial", alpha=1,  
#                              intercept=TRUE, standardize=FALSE)
    
    y.train.hat       =     predict(lasso.fit, newx = X.train, type = "class")
    y.test.hat        =     predict(lasso.fit, newx = X.test, type = "class")
    
    if (i == learn.pct1) {
      Err.lasso[s,1]    =     mean(y.train != y.train.hat)
      Err.lasso[s,3]    =     mean(y.test != y.test.hat)
      Err.lasso[s,5]    =     min(lasso.cv$cvm)
      
      Err.lasso[s,7]    =     mean(1 == y.train.hat[y.train==0]) # false positive
      Err.lasso[s,8]    =     mean(0 == y.train.hat[y.train==1]) # false negative
      Err.lasso[s,9]    =     mean(1 == y.test.hat[y.test==0]) # false positive
      Err.lasso[s,10]   =     mean(0 == y.test.hat[y.test==1]) # false negative
    } else {
      Err.lasso[s,2]    =     mean(y.train != y.train.hat)
      Err.lasso[s,4]    =     mean(y.test != y.test.hat)
      Err.lasso[s,6]    =     min(lasso.cv$cvm)
      
      Err.lasso[s,11]   =     mean(1 == y.train.hat[y.train==0]) # false positive
      Err.lasso[s,12]   =     mean(0 == y.train.hat[y.train==1]) # false negative
      Err.lasso[s,13]   =     mean(1 == y.test.hat[y.test==0]) # false positive
      Err.lasso[s,14]   =     mean(0 == y.test.hat[y.test==1]) # false negative
    }
    
    # optimize ridge logistic regression using cross validation
    m                 =     25
    ridge.cv          =     cv.glmnet(X.os.train, y.os.train, family = "binomial", alpha = 0, nfolds = 10, type.measure="class")
    ridge.fit         =     glmnet(X.os.train, y.os.train, lambda = ridge.cv$lambda.min, family = "binomial", alpha = 0)

#    ridge.cv      =    cv.glmnet(X.os.train, y.os.train, family="binomial", alpha=0, 
#                                 nfolds = 10, type.measure="class")
#    lam.ridge     =    exp(seq(log(max(ridge.cv$lambda)),log(0.00001), 
#                               -(log(max(ridge.cv$lambda))-log(0.00001))/(m-1)))
#    ridge.cv      =    cv.glmnet(X.os.train, y.os.train, lambda=lam.ridge, 
#                                 family="binomial", alpha=0,
#                                 nfolds=10, type.measure="class")
#    ridge.fit     =    glmnet(X.os.train, y.os.train, lambda=ridge.cv$lambda.min, 
#                              family="binomial", alpha=0)
    
#    ridge.cv      =    cv.glmnet(X.os.train, y.os.train, family="binomial", alpha=0, 
#                                 intercept=TRUE, standardize=FALSE, 
#                                 nfolds = 10, type.measure="class")
#    lam.ridge     =    exp(seq(log(max(ridge.cv$lambda)),log(0.00001), 
#                               -(log(max(ridge.cv$lambda))-log(0.00001))/(m-1)))
#    ridge.cv      =    cv.glmnet(X.os.train, y.os.train, lambda=lam.ridge, 
#                                 family="binomial", alpha=0,
#                                 intercept=TRUE, standardize=FALSE, 
#                                 nfolds=10, type.measure="class")
#    ridge.fit     =    glmnet(X.os.train, y.os.train, lambda=ridge.cv$lambda.min, 
#                              family="binomial", alpha=0,  
#                              intercept=TRUE, standardize=FALSE)
    
    y.train.hat       =     predict(ridge.fit, newx = X.train, type = "class")
    y.test.hat        =     predict(ridge.fit, newx = X.test, type = "class")
    
    if (i == learn.pct1) {
      Err.ridge[s,1]    =     mean(y.train != y.train.hat)
      Err.ridge[s,3]    =     mean(y.test != y.test.hat)
      Err.ridge[s,5]    =     min(ridge.cv$cvm)

      Err.ridge[s,7]    =     mean(1 == y.train.hat[y.train==0]) # false positive
      Err.ridge[s,8]    =     mean(0 == y.train.hat[y.train==1]) # false negative
      Err.ridge[s,9]    =     mean(1 == y.test.hat[y.test==0]) # false positive
      Err.ridge[s,10]   =     mean(0 == y.test.hat[y.test==1]) # false negative
      
    } else {
      Err.ridge[s,2]    =     mean(y.train != y.train.hat)
      Err.ridge[s,4]    =     mean(y.test != y.test.hat)
      Err.ridge[s,6]    =     min(ridge.cv$cvm)
      
      Err.ridge[s,11]   =     mean(1 == y.train.hat[y.train==0]) # false positive
      Err.ridge[s,12]   =     mean(0 == y.train.hat[y.train==1]) # false negative
      Err.ridge[s,13]   =     mean(1 == y.test.hat[y.test==0]) # false positive
      Err.ridge[s,14]   =     mean(0 == y.test.hat[y.test==1]) # false negative
    } 
    
    # output time
    ptm       =     proc.time() - ptm
    if (i == learn.pct1) {
      time1   =     ptm["elapsed"]
    } else {
      time2   =     ptm["elapsed"]
    } 
    
  }
  
  cat(sprintf("s=%1.f: 
              %.1fn | time: %0.3f(sec) |
              Train:  rf=%.2f, svm=%.2f, logistic=%.2f, lasso=%.2f, ridge=%.2f
              Test:   rf=%.2f, svm=%.2f, logistic=%.2f, lasso=%.2f, ridge=%.2f
              Min CV: rf=%.2f, svm=%.2f, logistic=%.2f, lasso=%.2f, ridge=%.2f
              %.1fn | time: %0.3f(sec) |
              Train:  rf=%.2f, svm=%.2f, logistic=%.2f, lasso=%.2f, ridge=%.2f
              Test:   rf=%.2f, svm=%.2f, logistic=%.2f, lasso=%.2f, ridge=%.2f
              Min CV: rf=%.2f, svm=%.2f, logistic=%.2f, lasso=%.2f, ridge=%.2f\n",s,
              learn.pct1,time1,
              Err.rf[s,1],Err.svm[s,1],Err.logistic[s,1],Err.lasso[s,1],Err.ridge[s,1],
              Err.rf[s,3],Err.svm[s,3],Err.logistic[s,3],Err.lasso[s,3],Err.ridge[s,3],
              Err.rf[s,5],Err.svm[s,5],Err.logistic[s,5],Err.lasso[s,5],Err.ridge[s,5],
              learn.pct2,time2,
              Err.rf[s,2],Err.svm[s,2],Err.logistic[s,2],Err.lasso[s,2],Err.ridge[s,2],
              Err.rf[s,4],Err.svm[s,4],Err.logistic[s,4],Err.lasso[s,4],Err.ridge[s,4],
              Err.rf[s,6],Err.svm[s,6],Err.logistic[s,6],Err.lasso[s,6],Err.ridge[s,6]))
  
}

# save sample order
write.csv(data.frame(random_order), 
          "D:/d/Courses/STA/STA 9891/Project/random_order.csv", row.names=FALSE)

# Err is S x 14 matrix
# column  1 of Err = total train error (n learn = 0.5n)
# column  2 of Err = total train error (n learn = 0.9n)

# column  3 of Err = total test error (n learn = 0.5n)
# column  4 of Err = total test error (n learn = 0.9n)

# column  5 of Err = min cv error (n learn = 0.5n)
# column  6 of Err = min cv error (n learn = 0.9n)

# column  7 of Err = false positive train error (n learn = 0.5n)
# column  8 of Err = false negative train error (n learn = 0.5n)
# column  9 of Err = false positive test error (n learn = 0.5n)
# column 10 of Err = false negative test error (n learn = 0.5n)

# column 11 of Err = false positive train error (n learn = 0.9n)
# column 12 of Err = false negative train error (n learn = 0.9n)
# column 13 of Err = false positive test error (n learn = 0.9n)
# column 14 of Err = false negative test error (n learn = 0.9n)

# save files
write.csv(data.frame(Err.rf), 
          "D:/d/Courses/STA/STA 9891/Project/Err.rf.csv", row.names=FALSE)
write.csv(data.frame(Err.svm), 
          "D:/d/Courses/STA/STA 9891/Project/Err.svm.csv", row.names=FALSE)
write.csv(data.frame(Err.logistic), 
          "D:/d/Courses/STA/STA 9891/Project/Err.logistic.csv", row.names=FALSE)
write.csv(data.frame(Err.lasso), 
          "D:/d/Courses/STA/STA 9891/Project/Err.lasso.csv", row.names=FALSE)
write.csv(data.frame(Err.ridge), 
          "D:/d/Courses/STA/STA 9891/Project/Err.ridge.csv", row.names=FALSE)

# 0.5n train size
err.train.pct1 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                              rep("logistic lasso",S),rep("logistic ridge",S)), 
                            c(Err.rf[,1],Err.svm[,1],Err.logistic[,1],
                              Err.lasso[,1],Err.ridge[,1]))

err.test.pct1  = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                              rep("logistic lasso",S),rep("logistic ridge",S)), 
                            c(Err.rf[,3],Err.svm[,3],Err.logistic[,3],
                              Err.lasso[,3],Err.ridge[,3]))

err.minCV.pct1 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                              rep("logistic lasso",S),rep("logistic ridge",S)), 
                            c(Err.rf[,5],Err.svm[,5],Err.logistic[,5],
                              Err.lasso[,5],Err.ridge[,5]))

# train false positive (fp) - 0.5n
err.train.fp.pct1 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                                 rep("logistic lasso",S),rep("logistic ridge",S)), 
                               c(Err.rf[,7],Err.svm[,7],Err.logistic[,7],
                                 Err.lasso[,7],Err.ridge[,7]))
# train false negative (fn) - 0.5n
err.train.fn.pct1 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                                 rep("logistic lasso",S),rep("logistic ridge",S)), 
                               c(Err.rf[,8],Err.svm[,8],Err.logistic[,8],
                                 Err.lasso[,8],Err.ridge[,8]))

# test false positive (fp) - 0.5n
err.test.fp.pct1 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                                rep("logistic lasso",S),rep("logistic ridge",S)), 
                              c(Err.rf[,9],Err.svm[,9],Err.logistic[,9],
                                Err.lasso[,9],Err.ridge[,9]))
# train false negative (fn) - 0.5n
err.test.fn.pct1 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                                rep("logistic lasso",S),rep("logistic ridge",S)), 
                              c(Err.rf[,10],Err.svm[,10],Err.logistic[,10],
                                Err.lasso[,10],Err.ridge[,10]))

# 0.9n train size
err.train.pct2 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                              rep("logistic lasso",S),rep("logistic ridge",S)), 
                            c(Err.rf[,2],Err.svm[,2],Err.logistic[,2],
                              Err.lasso[,2],Err.ridge[,2]))

err.test.pct2  = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                              rep("logistic lasso",S),rep("logistic ridge",S)), 
                            c(Err.rf[,4],Err.svm[,4],Err.logistic[,4],
                              Err.lasso[,4],Err.ridge[,4]))

err.minCV.pct2 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                              rep("logistic lasso",S),rep("logistic ridge",S)), 
                            c(Err.rf[,6],Err.svm[,6],Err.logistic[,6],
                              Err.lasso[,6],Err.ridge[,6]))

# train false positive (fp) - 0.9n
err.train.fp.pct2 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                                 rep("logistic lasso",S),rep("logistic ridge",S)), 
                               c(Err.rf[,11],Err.svm[,11],Err.logistic[,11],
                                 Err.lasso[,11],Err.ridge[,11]))
# train false negative (fn) - 0.9n
err.train.fn.pct2 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                                 rep("logistic lasso",S),rep("logistic ridge",S)), 
                               c(Err.rf[,12],Err.svm[,12],Err.logistic[,12],
                                 Err.lasso[,12],Err.ridge[,12]))

# test false positive (fp) - 0.9n
err.test.fp.pct2 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                                rep("logistic lasso",S),rep("logistic ridge",S)), 
                              c(Err.rf[,13],Err.svm[,13],Err.logistic[,13],
                                Err.lasso[,13],Err.ridge[,13]))
# train false negative (fn) - 0.9n
err.test.fn.pct2 = data.frame(c(rep("rf",S),rep("radial svm",S),rep("logistic",S),
                                rep("logistic lasso",S),rep("logistic ridge",S)), 
                              c(Err.rf[,14],Err.svm[,14],Err.logistic[,14],
                                Err.lasso[,14],Err.ridge[,14]))

colnames(err.train.pct1)    =     c("method","err")
colnames(err.test.pct1)     =     c("method","err")
colnames(err.minCV.pct1)    =     c("method","err")
colnames(err.train.pct2)    =     c("method","err")
colnames(err.test.pct2)     =     c("method","err")
colnames(err.minCV.pct2)    =     c("method","err")

colnames(err.train.fp.pct1) =     c("method","err")
colnames(err.train.fn.pct1) =     c("method","err")
colnames(err.test.fp.pct1)  =     c("method","err")
colnames(err.test.fn.pct1)  =     c("method","err")

colnames(err.train.fp.pct2) =     c("method","err")
colnames(err.train.fn.pct2) =     c("method","err")
colnames(err.test.fp.pct2)  =     c("method","err")
colnames(err.test.fn.pct2)  =     c("method","err")

# save files
write.csv(err.train.pct1, 
          "D:/d/Courses/STA/STA 9891/Project/err.train.pct1.csv", 
          row.names=FALSE)
write.csv(err.test.pct1, 
          "D:/d/Courses/STA/STA 9891/Project/err.test.pct1.csv", 
          row.names=FALSE)
write.csv(err.minCV.pct1, 
          "D:/d/Courses/STA/STA 9891/Project/err.minCV.pct1.csv", 
          row.names=FALSE)
write.csv(err.train.pct2, 
          "D:/d/Courses/STA/STA 9891/Project/err.train.pct2.csv", 
          row.names=FALSE)
write.csv(err.test.pct2, 
          "D:/d/Courses/STA/STA 9891/Project/err.test.pct2.csv", 
          row.names=FALSE)
write.csv(err.minCV.pct2, 
          "D:/d/Courses/STA/STA 9891/Project/err.minCV.pct2.csv", 
          row.names=FALSE)
write.csv(err.train.fp.pct1, 
          "D:/d/Courses/STA/STA 9891/Project/err.train.fp.pct1.csv", 
          row.names=FALSE)
write.csv(err.train.fn.pct1, 
          "D:/d/Courses/STA/STA 9891/Project/err.train.fn.pct1.csv", 
          row.names=FALSE)
write.csv(err.test.fp.pct1, 
          "D:/d/Courses/STA/STA 9891/Project/err.test.fp.pct1.csv", 
          row.names=FALSE)
write.csv(err.test.fn.pct1, 
          "D:/d/Courses/STA/STA 9891/Project/err.test.fn.pct1.csv", 
          row.names=FALSE)
write.csv(err.train.fp.pct2, 
          "D:/d/Courses/STA/STA 9891/Project/err.train.fp.pct2.csv", 
          row.names=FALSE)
write.csv(err.train.fn.pct2, 
          "D:/d/Courses/STA/STA 9891/Project/err.train.fn.pct2.csv", 
          row.names=FALSE)
write.csv(err.test.fp.pct2, 
          "D:/d/Courses/STA/STA 9891/Project/err.test.fp.pct2.csv", 
          row.names=FALSE)
write.csv(err.test.fn.pct2, 
          "D:/d/Courses/STA/STA 9891/Project/err.test.fn.pct2.csv", 
          row.names=FALSE)

####################
## 4.(b) boxplots ##
####################
# Method 1: Generate 6 boxplots with 6 legend for each n learn
p1 = ggplot(err.train.pct1)   +   aes(x=method, y = err, fill=method) +   geom_boxplot()  +
  ggtitle("0.5n train errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p2 = ggplot(err.test.pct1)   +    aes(x=method, y = err, fill=method) +   geom_boxplot()  +
  ggtitle("0.5n test errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p3 = ggplot(err.minCV.pct1)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  +  
  ggtitle("0.5n min CV errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p4 = ggplot(err.train.pct2)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  +  
  ggtitle("0.9n train errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p5 = ggplot(err.test.pct2)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  +  
  ggtitle("0.9n test errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p6 = ggplot(err.minCV.pct2)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
  ggtitle("0.9n min CV errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p7 = ggplot(err.train.fp.pct1)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
  ggtitle("0.5n train fp errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p8 = ggplot(err.train.fn.pct1)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
  ggtitle("0.5n train fn errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p9 = ggplot(err.test.fp.pct1)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
  ggtitle("0.5n test fp errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p10 = ggplot(err.test.fn.pct1)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
  ggtitle("0.5n test fn errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p11 = ggplot(err.train.fp.pct2)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
  ggtitle("0.9n train fp errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p12 = ggplot(err.train.fn.pct2)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
  ggtitle("0.9n train fn errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p13 = ggplot(err.test.fp.pct2)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
  ggtitle("0.9n test fp errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

p14 = ggplot(err.test.fn.pct2)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  + 
  ggtitle("0.9n test fp errors") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=45, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=45, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 1)  

grid.arrange(p1,  p7,  p8, p2,  p9, p10, ncol=3)
grid.arrange(p4, p11, p12, p5, p13, p14, ncol=3)

# Method 2: Generate 4 boxplots with only one legend
giant_df = data.frame(c(rep("0.5n train errors",5*S), 
                        rep("0.5n test errors",5*S), 
                        rep("0.9n train errors",5*S), 
                        rep("0.9n test errors",5*S)), 
                      
                      rbind(err.train.pct1, 
                            err.test.pct1, 
                            err.train.pct2, 
                            err.test.pct2) 
                      ) 

colnames(giant_df)    =     c("error_type","method","err")
giant_df$error_type = factor(giant_df$error_type, 
                             levels=c("0.5n train errors", 
                                      "0.5n test errors", 
                                      "0.9n train errors", 
                                      "0.9n test errors"))

giant_p = ggplot(giant_df)   +     
  aes(x=method, y = err, fill=method) +   
  geom_boxplot()  + 
  facet_wrap(~ error_type, ncol=2) + 
  ggtitle("Boxplots of Error Rates (train size = 0.5n, 0.9n, 100 samples)") +
  theme( axis.title.x = element_text(size =12, face  = "bold", family = "Courier"),
         plot.title   = element_text(size =12, family= "Courier"), 
         axis.title.y = element_text(size =12, face  = "bold", family = "Courier"), 
         axis.text.x  = element_text(angle=10, hjust =  1, size=10, face="bold", family="Courier"), 
         axis.text.y  = element_text(angle=10, vjust =0.7, size=10, face="bold", family="Courier"))+
  ylim(0, 0.6)  

giant_p 

###########
## 4.(c) ##
###########
######################################################
## 4.(c) 10-fold CV curves for lasso, ridge and svm ##
######################################################
s = 54 # can choose the sample order as we want
set.seed(1)

for (j in 1:2) {
  
  n.train      =  floor(n*learn.pct[j])
  n.test       =  n-n.train
  trainSet     =  random_order[s,][1:n.train]
  testSet      =  random_order[s,][(1+n.train):n] 
  X.train      =  X[trainSet, ]
  y.train      =  y[trainSet]
  X.test       =  X[testSet, ]
  y.test       =  y[testSet]
  y.os.train    =  y.train   # initialize the over-sampled (os) set to train the models
  X.os.train    =  X.train   # initialize the over-sampled (os) set to train the models
  
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # to take into account the imbalance 
  # below we over-sample (with replacement) the  data so that the data is balanced
  imbalance     =     FALSE   
  if (imbalance == TRUE) {
    index.yis0      =      which(y.train==0)  # idetify the index of  points with label 0
    index.yis1      =      which(y.train==1) # idetify the index of  points with label 1
    n.train.1       =      length(index.yis1)
    n.train.0       =      length(index.yis0)
    if (n.train.1 > n.train.0) {     # we need more 0s in out training set, so we over sample with replacement
      more.train    =      sample(index.yis0, size=n.train.1-n.train.0, replace=TRUE)
    }         else {    # we need more 1s in out training set, so we over sample with replacement          
      more.train    =      sample(index.yis1, size=n.train.0-n.train.1, replace=TRUE)
    }
    ##### the code below CORRECTLY over samples the train set 
    ##### and stores it in y.train_ and X.train_
    y.os.train        =       as.factor(c(y.train, y.train[more.train])-1) 
    X.os.train        =       rbind2(X.train, X.train[more.train,])    
    
    ##### the code below is MISTAKENLY over sample the train set  
    # trainSet       =       rbind(trainSet, more.train)
    # y.train        =       y[trainSet]
    # X.train        =       X[trainSet, ]
    # n.train        =       dim(X.train)[1] 
    ##################################################################################
  } # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  # alternative way of breaking data into train and test 
  #    train.data              =   data.frame(X[trainSet, ], as.factor(y[trainSet]))
  #    test.data               =   data.frame(X[testSet, ], as.factor(y[testSet]))
  #    names(train.data)[89]   =   "target"
  #    names(test.data)[89]    =   "target"
  
  os.train.data           =   data.frame(X.os.train, as.factor(y.os.train))
  train.data              =   data.frame(X.train, as.factor(y.train))
  test.data               =   data.frame(X.test, as.factor(y.test))
  names(os.train.data)[89]=   "target"
  names(train.data)[89]   =   "target"
  names(test.data)[89]    =   "target"
  
  m             =    25
  
  # lasso cv curve and time
  lasso.cv      =    cv.glmnet(X.os.train, y.os.train, family="binomial", alpha=1, 
                               intercept=TRUE, standardize=FALSE,  
                               nfolds=10, type.measure="class")
  lam.lasso     =    exp(seq(log(max(lasso.cv$lambda)),log(0.00001), 
                             (log(0.00001) - log(max(lasso.cv$lambda)))/(m-1)))
  
  ptm           =    proc.time()
  lasso.cv      =    cv.glmnet(X.os.train, y.os.train, lambda=lam.lasso, 
                               family="binomial", alpha=1, intercept=TRUE, 
                               standardize=FALSE, nfolds=10, type.measure="class")
  ptm           =    proc.time() - ptm
  time.lasso.cv =    ptm["elapsed"] 
  
  ptm           =    proc.time()
  lasso.fit     =    glmnet(X.os.train, y.os.train, lambda=lasso.cv$lambda, 
                            family="binomial", alpha=1, 
                            intercept=TRUE, standardize=FALSE)
  ptm           =    proc.time() - ptm
  time.lasso.fit=    ptm["elapsed"] 
  
  lasso.fit.0   =    glmnet(X.os.train, y.os.train, lambda=0, 
                            family="binomial", alpha=1, 
                            intercept=TRUE, standardize=FALSE)
  
  n.lambdas     =    dim(lasso.fit$beta)[2]
  lasso.beta.ratio    =    rep(0, n.lambdas)
  for (i in 1:n.lambdas) {
    lasso.beta.ratio[i] = sum(abs(lasso.fit$beta[,i]))/sum(abs(lasso.fit.0$beta))
  }
  
  # ridge cv curve and time
  ridge.cv      =    cv.glmnet(X.os.train, y.os.train, family="binomial", alpha=0, 
                               intercept=TRUE, standardize=FALSE, 
                               nfolds = 10, type.measure="class")
  lam.ridge     =    exp(seq(log(max(ridge.cv$lambda)),log(0.00001), 
                             -(log(max(ridge.cv$lambda))-log(0.00001))/(m-1)))
  
  ptm           =    proc.time()
  ridge.cv      =    cv.glmnet(X.os.train, y.os.train, lambda=lam.ridge, 
                               family="binomial", alpha=0, intercept=TRUE, 
                               standardize=FALSE, nfolds=10, type.measure="class")
  ptm           =    proc.time() - ptm
  time.ridge.cv =    ptm["elapsed"] 
  
  ptm           =    proc.time()
  ridge.fit     =    glmnet(X.os.train, y.os.train, lambda=ridge.cv$lambda, 
                            family="binomial", alpha=0,  
                            intercept=TRUE, standardize=FALSE)
  ptm           =    proc.time() - ptm
  time.ridge.fit=    ptm["elapsed"] 
  
  ridge.fit.0   =    glmnet(X.os.train, y.os.train, lambda=0, 
                            family="binomial", alpha=0, 
                            intercept=TRUE, standardize=FALSE)
  
  n.lambdas     =    dim(ridge.fit$beta)[2]
  ridge.beta.ratio    =    rep(0, n.lambdas)
  for (i in 1:n.lambdas) {
    ridge.beta.ratio[i] = sqrt(sum((ridge.fit$beta[,i])^2)/sum((ridge.fit.0$beta)^2))
  }
  
  # Plot CV curves for lasso and ridge on one plot
  eror           =     data.frame(c(rep("lasso", length(lasso.beta.ratio)), 
                                    rep("ridge", length(ridge.beta.ratio)) ), 
                                  c(lasso.beta.ratio, ridge.beta.ratio) ,
                                  c(lasso.cv$cvm, ridge.cv$cvm),
                                  c(lasso.cv$cvsd, ridge.cv$cvsd))
  colnames(eror) =     c("method", "ratio", "cv", "sd")
  
  eror.plot      =     ggplot(eror, aes(x=ratio, y = cv, color=method)) +   geom_line(size=1) 
  eror.plot      =     eror.plot  + theme(legend.text = element_text(colour="black", size=16, face="bold", family = "Courier")) 
  eror.plot      =     eror.plot  + geom_pointrange(aes(ymin=cv-sd, ymax=cv+sd),  size=0.8,  shape=15)
  eror.plot      =     eror.plot  + theme(legend.title=element_blank()) 
  eror.plot      =     eror.plot  + scale_color_discrete(breaks=c("lasso", "ridge"))
  eror.plot      =     eror.plot  + theme(axis.title.x = element_text(size=24),
                                          axis.text.x  = element_text(angle=0, vjust=0.5, size=14),
                                          axis.text.y  = element_text(angle=0, vjust=0.5, size=14)) 
  #eror.plot      =     eror.plot  + theme(axis.title.y = element_text(size=16, face="bold", family = "Courier")) 
  #eror.plot      =     eror.plot  + xlab( expression(paste( lambda))) + ylab("")
  eror.plot      =     eror.plot  + theme(plot.title = element_text(hjust = 0.5, vjust = -10, size=20, family = "Courier"))
  #eror.plot      =     eror.plot  + ggtitle(TeX(sprintf("$n$=%s,$p$=%s,$t_{LO}$=%s,$t_{ALO}$=%0.3f,$t_{FIT}$=%.3f",n,p,time.lo,time.alo,time.fit))) 
  eror.plot      =     eror.plot  + ggtitle((sprintf("%.1fn: \n lasso.cv:%0.3f(sec), lasso.fit:%0.3f(sec) \n ridge.cv:%0.3f(sec), ridge.fit:%0.3f(sec)",learn.pct[j],time.lasso.cv,time.lasso.fit,time.ridge.cv,time.ridge.fit))) 
  
  if (j == 1) {
    eror.plot.1 =   eror.plot 
    time.lasso.1 =   time.lasso.cv + time.lasso.fit    
    time.ridge.1 =   time.ridge.cv + time.ridge.fit
  } else {
    eror.plot.2 =   eror.plot 
    time.lasso.2 =   time.lasso.cv + time.lasso.fit
    time.ridge.2 =   time.ridge.cv + time.ridge.fit
  }
  
  
  # SVM CV error and time
  ptm         =   proc.time()
  tune.svm    =   tune(svm, target ~ ., data=train.data, kernel = "radial", 
                       ranges = list(cost = 10^seq(-2,2,length.out = 5), 
                                     gamma = 10^seq(-2,2,length.out = 5) ))
  ptm         =   proc.time() - ptm
  time.svm.cv =   ptm["elapsed"] 
  
  ptm         =   proc.time()
  svm.fit     =   svm(target~., data=train.data, kernel="radial", 
                      cost=tune.svm$performances$cost, 
                      gamma=tune.svm$performances$gamma)
  ptm         =   proc.time() - ptm
  time.svm.fit=   ptm["elapsed"] 
  
  # Generate SVM CV error heatmap
  svm.df   = data.frame(as.character(tune.svm$performances[,1]), 
                        as.character(tune.svm$performances[,2]), 
                        tune.svm$performances[,3])
  colnames(svm.df) = c("cost", "gamma","error")
  
  svm.heat = ggplot(svm.df, aes(gamma, cost, fill=error)) + 
    geom_tile(colour = "white") + 
    geom_text(aes(label = round(error, 3))) +
    scale_fill_gradientn(colours=c("yellow", "red")) + 
    ggtitle((sprintf("%.1fn: \n svm.cv:%0.3f(sec), svm.fit:%0.3f(sec)",learn.pct[j],time.svm.cv,time.svm.fit))) + 
    theme_classic() 
  
  if (j == 1) {
    svm.heat.1 =   svm.heat 
    time.svm.1 =   time.svm.cv + time.svm.fit
  } else {
    svm.heat.2 =   svm.heat
    time.svm.2 =   time.svm.cv + time.svm.fit
  }
  
  # record random forest cv time and fitting time
  ptm         =   proc.time()
  tune.rf     =   tune(randomForest, target ~ ., data=os.train.data)
  ptm         =   proc.time() - ptm
  time.rf.cv  =   ptm["elapsed"] 
  
  if (j == 1) {
    time.rf.cv.1 =   time.rf.cv 
  } else {
    time.rf.cv.2 =   time.rf.cv 
  }
  
  ptm         =   proc.time()
  rf.fit      =   randomForest(target~., data=os.train.data, mtry=sqrt(p), 
                               importance=TRUE)
  ptm         =   proc.time() - ptm
  time.rf.fit =   ptm["elapsed"] 
  
  if (j == 1) {
    time.rf.fit.1 =   time.rf.fit
  } else {
    time.rf.fit.2 =   time.rf.fit
  }
  
  
} 

eror.plot.1
eror.plot.2

svm.heat.1
svm.heat.2

time.rf.cv.1
time.rf.cv.2
time.rf.fit.1
time.rf.fit.2 

########################################################################
## 4.(c) Model performance vs. fitting time for lasso, ridge, RF, SVM ##
########################################################################
# create performance vs. fitting time data.frame
pt.df = data.frame(c("Lasso", "Ridge", "RF", "SVM", "Lasso", "Ridge", "RF", "SVM"), 
                   c(rep("0.5n", 4), rep("0.9n", 4)), 
                   c(time.lasso.1, time.ridge.1, time.rf.fit.1, time.svm.1,
                     time.lasso.2, time.ridge.2, time.rf.fit.2, time.svm.2), 
                   c(Err.lasso[s,3], Err.ridge[s,3], Err.rf[s,3], Err.svm[s,3], 
                     Err.lasso[s,4], Err.ridge[s,4], Err.rf[s,4], Err.svm[s,4])
)
colnames(pt.df) = c("model", "train_size", "fitting_time", "test_error")
pt.df

write.csv(pt.df, 
          "D:/d/Courses/STA/STA 9891/Project/performance_vs_time.csv", 
          row.names=FALSE)

#pt.df = read.csv("D:/d/Courses/STA/STA 9891/Project/performance_vs_time.csv",header=TRUE)
#colnames(pt.df) = c("model", "train_size", "fitting_time", "test_error")
#pt.df

ggplot(pt.df, aes(x=fitting_time, y=test_error, color=model)) +
  geom_point(size=5, alpha=0.7) + 
  facet_wrap(~ train_size, nrow=2) + 
  labs(x="Fitting Time (sec)") + 
  labs(y="Performance (Test Error Rate)") + 
  ylim(0, 0.6) + 
  ggtitle("Performance vs. Fitting Time (0.5n, 0.9n)")


#################################################################
## 4.(d) bar-plots of the estimated coecients (lasso, ridge), ##
##       the importance of the parameters, for each nlearn     ##
#################################################################
s = 54 # can choose the sample order as we want
set.seed(1)

for (j in 1:2) {
  
  n.train      =  floor(n*learn.pct[j])
  n.test       =  n-n.train
  trainSet     =  random_order[s,][1:n.train]
  testSet      =  random_order[s,][(1+n.train):n] 
  X.train      =  X[trainSet, ]
  y.train      =  y[trainSet]
  X.test       =  X[testSet, ]
  y.test       =  y[testSet]
  y.os.train    =  y.train   # initialize the over-sampled (os) set to train the models
  X.os.train    =  X.train   # initialize the over-sampled (os) set to train the models
  
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # to take into account the imbalance 
  # below we over-sample (with replacement) the  data so that the data is balanced
  imbalance     =     FALSE   
  if (imbalance == TRUE) {
    index.yis0      =      which(y.train==0)  # idetify the index of  points with label 0
    index.yis1      =      which(y.train==1) # idetify the index of  points with label 1
    n.train.1       =      length(index.yis1)
    n.train.0       =      length(index.yis0)
    if (n.train.1 > n.train.0) {     # we need more 0s in out training set, so we over sample with replacement
      more.train    =      sample(index.yis0, size=n.train.1-n.train.0, replace=TRUE)
    }         else {    # we need more 1s in out training set, so we over sample with replacement          
      more.train    =      sample(index.yis1, size=n.train.0-n.train.1, replace=TRUE)
    }
    ##### the code below CORRECTLY over samples the train set 
    ##### and stores it in y.train_ and X.train_
    y.os.train        =       as.factor(c(y.train, y.train[more.train])-1) 
    X.os.train        =       rbind2(X.train, X.train[more.train,])    
    
    ##### the code below is MISTAKENLY over sample the train set  
    # trainSet       =       rbind(trainSet, more.train)
    # y.train        =       y[trainSet]
    # X.train        =       X[trainSet, ]
    # n.train        =       dim(X.train)[1] 
    ##################################################################################
  } # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  # alternative way of breaking data into train and test 
  #    train.data              =   data.frame(X[trainSet, ], as.factor(y[trainSet]))
  #    test.data               =   data.frame(X[testSet, ], as.factor(y[testSet]))
  #    names(train.data)[89]   =   "target"
  #    names(test.data)[89]    =   "target"
  
  os.train.data           =   data.frame(X.os.train, as.factor(y.os.train))
  train.data              =   data.frame(X.train, as.factor(y.train))
  test.data               =   data.frame(X.test, as.factor(y.test))
  names(os.train.data)[89]=   "target"
  names(train.data)[89]   =   "target"
  names(test.data)[89]    =   "target"
  
  # lasso barplots 
  lasso.cv      =    cv.glmnet(X.os.train, y.os.train, family="binomial", alpha=1, 
                               intercept=TRUE, standardize=FALSE,  
                               nfolds=10, type.measure="class")
  lam.lasso     =    exp(seq(log(max(lasso.cv$lambda)),log(0.00001), 
                             (log(0.00001) - log(max(lasso.cv$lambda)))/(m-1)))
  
  lasso.cv      =    cv.glmnet(X.os.train, y.os.train, lambda=lam.lasso, 
                               family="binomial", alpha=1, intercept=TRUE, 
                               standardize=FALSE, nfolds=10, type.measure="class")
  
  lasso.fit     =    glmnet(X.os.train, y.os.train, lambda=lasso.cv$lambda.min, 
                            family="binomial", alpha=1, 
                            intercept=TRUE, standardize=FALSE)
  
  lasso.df = data.frame(Predictor=as.vector(row.names(lasso.fit$beta)), 
                        lasso.coef=as.vector(lasso.fit$beta))
  
  if (j == 1) {
    lasso.bar = ggplot(lasso.df, aes(x=Predictor, y=lasso.coef)) + 
      geom_bar(stat="identity", fill="#F8766D")+ 
      labs(x="Predictor") + 
      labs(y="Lasso Coefficient") + 
      ylim(-0.75, 0.75) + 
      coord_flip() + 
      ggtitle((sprintf("%.1fn Lasso Coefficient",learn.pct[j]))) 
    
    lasso.bar_10 = top_n(lasso.df, n=10, abs(lasso.coef)) %>% 
      ggplot(., aes(x=reorder(Predictor, lasso.coef), y=lasso.coef)) + 
      geom_bar(stat="identity", fill="#F8766D")+ 
      labs(x="Predictor") + 
      labs(y="Lasso Coefficient") + 
      ylim(-0.75, 0.75) + 
      coord_flip() + 
      ggtitle((sprintf("%.1fn Lasso Coefficient (Top 10)",learn.pct[j]))) 
  } else {
    lasso.bar = ggplot(lasso.df, aes(x=Predictor, y=lasso.coef)) + 
      geom_bar(stat="identity", fill="#619CFF")+ 
      labs(x="Predictor") + 
      labs(y="Lasso Coefficient") + 
      ylim(-0.75, 0.75) + 
      coord_flip() + 
      ggtitle((sprintf("%.1fn Lasso Coefficient",learn.pct[j]))) 
    
    lasso.bar_10 = top_n(lasso.df, n=10, abs(lasso.coef)) %>% 
      ggplot(., aes(x=reorder(Predictor, lasso.coef), y=lasso.coef)) + 
      geom_bar(stat="identity", fill="#619CFF")+ 
      labs(x="Predictor") + 
      labs(y="Lasso Coefficient") + 
      ylim(-0.75, 0.75) + 
      coord_flip() + 
      ggtitle((sprintf("%.1fn Lasso Coefficient (Top 10)",learn.pct[j]))) 
  }
  
  # ridge barplots 
  ridge.cv      =    cv.glmnet(X.os.train, y.os.train, family="binomial", alpha=0, 
                               intercept=TRUE, standardize=FALSE, 
                               nfolds = 10, type.measure="class")
  lam.ridge     =    exp(seq(log(max(ridge.cv$lambda)),log(0.00001), 
                             -(log(max(ridge.cv$lambda))-log(0.00001))/(m-1)))
  
  ridge.cv      =    cv.glmnet(X.os.train, y.os.train, lambda=lam.ridge, 
                               family="binomial", alpha=0, intercept=TRUE, 
                               standardize=FALSE, nfolds=10, type.measure="class")
  
  ridge.fit     =    glmnet(X.os.train, y.os.train, lambda=ridge.cv$lambda.min, 
                            family="binomial", alpha=0,  
                            intercept=TRUE, standardize=FALSE)
  
  ridge.df = data.frame(Predictor=as.vector(row.names(ridge.fit$beta)), 
                        ridge.coef=as.vector(ridge.fit$beta))
  
  if (j == 1) {
    ridge.bar = ggplot(ridge.df, aes(x=Predictor, y=ridge.coef)) + 
      geom_bar(stat="identity", fill="#F8766D")+ 
      labs(x="Predictor") + 
      labs(y="Ridge Coefficient") + 
      ylim(-0.75, 0.75) + 
      coord_flip() + 
      ggtitle((sprintf("%.1fn Ridge Coefficient",learn.pct[j]))) 
    
    ridge.bar_10 = top_n(ridge.df, n=10, abs(ridge.coef)) %>% 
      ggplot(., aes(x=reorder(Predictor, ridge.coef), y=ridge.coef)) + 
      geom_bar(stat="identity", fill="#F8766D")+ 
      labs(x="Predictor") + 
      labs(y="Ridge Coefficient") + 
      ylim(-0.75, 0.75) + 
      coord_flip() + 
      ggtitle((sprintf("%.1fn Ridge Coefficient (Top 10)",learn.pct[j]))) 
  } else {
    ridge.bar = ggplot(ridge.df, aes(x=Predictor, y=ridge.coef)) + 
      geom_bar(stat="identity", fill="#619CFF")+ 
      labs(x="Predictor") + 
      labs(y="Ridge Coefficient") + 
      ylim(-0.75, 0.75) + 
      coord_flip() + 
      ggtitle((sprintf("%.1fn Ridge Coefficient",learn.pct[j]))) 
    
    ridge.bar_10 = top_n(ridge.df, n=10, abs(ridge.coef)) %>% 
      ggplot(., aes(x=reorder(Predictor, ridge.coef), y=ridge.coef)) + 
      geom_bar(stat="identity", fill="#619CFF")+ 
      labs(x="Predictor") + 
      labs(y="Ridge Coefficient") + 
      ylim(-0.75, 0.75) + 
      coord_flip() + 
      ggtitle((sprintf("%.1fn Ridge Coefficient (Top 10)",learn.pct[j]))) 
  }
  
  # rf barplots
  rf.fit      =   randomForest(target~., data=os.train.data, mtry=sqrt(p), 
                               importance=TRUE)
  
  rf.df = data.frame(Predictor=c(row.names(rf.fit$importance)), rf.var.imp=c(rf.fit$importance[,4]))
  
  if (j == 1) {
    rf.bar = ggplot(rf.df, aes(x=Predictor, y=rf.var.imp)) + 
      geom_bar(stat="identity", fill="#F8766D")+ 
      labs(y="RF Var Importance") + ylim(0, 11.5) + 
      coord_flip() + 
      ggtitle((sprintf("%.1fn RF Variable Importance",learn.pct[j]))) 
    
    rf.bar_10 = top_n(rf.df, n=10, rf.var.imp) %>%
      ggplot(., aes(x=reorder(Predictor, rf.var.imp), y=rf.var.imp)) + 
      geom_bar(stat="identity", fill="#F8766D")+ 
      labs(y="RF Var Importance") + ylim(0, 11.5) +
      coord_flip() + 
      ggtitle((sprintf("%.1fn RF Variable Importance (Top 10)",learn.pct[j])))  
    
  } else {
    rf.bar = ggplot(rf.df, aes(x=Predictor, y=rf.var.imp)) + 
      geom_bar(stat="identity", fill="#619CFF")+ 
      labs(y="RF Var Importance") + ylim(0, 11.5) + 
      coord_flip() + 
      ggtitle((sprintf("%.1fn RF Variable Importance",learn.pct[j])))  
    
    rf.bar_10 = top_n(rf.df, n=10, rf.var.imp) %>%
      ggplot(., aes(x=reorder(Predictor, rf.var.imp), y=rf.var.imp)) + 
      geom_bar(stat="identity", fill="#619CFF")+ 
      labs(y="RF Var Importance") + ylim(0, 11.5) +
      coord_flip() + 
      ggtitle((sprintf("%.1fn RF Variable Importance (Top 10)",learn.pct[j])))  
    
  }
  
  
  if (j == 1) {
    lasso.bar1 = lasso.bar
    ridge.bar1 = ridge.bar
    rf.bar1    = rf.bar 
    
    lasso.bar_10_1 = lasso.bar_10
    ridge.bar_10_1 = ridge.bar_10
    rf.bar_10_1    = rf.bar_10 
  } else {
    lasso.bar2 = lasso.bar
    ridge.bar2 = ridge.bar
    rf.bar2    = rf.bar 
    
    lasso.bar_10_2 = lasso.bar_10
    ridge.bar_10_2 = ridge.bar_10
    rf.bar_10_2    = rf.bar_10 
  }
  
}

grid.arrange(lasso.bar1, ridge.bar1, rf.bar1, 
             lasso.bar2, ridge.bar2, rf.bar2, ncol=3)
grid.arrange(lasso.bar_10_1, ridge.bar_10_1, rf.bar_10_1, 
             lasso.bar_10_2, ridge.bar_10_2, rf.bar_10_2, ncol=3)
