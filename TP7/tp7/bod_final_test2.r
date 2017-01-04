# LOAD PACKAGES 
# =============
rm(list=ls())
library(MASS)
library(leaps)
library(glmnet)
library(class)
library(e1071)
library(tree)
library(randomForest)
library("neuralnet")
library(nnet)
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')


# LOAD DATA
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/end_5.rData")

qda.acp.error.k5 <- qda.acp.error
qda.lda.error.k5 <- qda.lda.error
qda.forward.error.k5 <- qda.forward.error
lda.acp.error.k5 <- lda.acp.error
lda.lda.error.k5 <- lda.lda.error
lda.forward.error.k5 <- lda.forward.error
knn.acp.error.k5 <- knn.acp.error
knn.lda.error.k5 <- knn.lda.error
knn.forward.error.k5 <- knn.forward.error
logReg.acp.error.k5 <- logReg.acp.error
logReg.lda.error.k5 <- logReg.lda.error
logReg.forward.error.k5 <- logReg.forward.error
nb.acp.error.k5 <- nb.acp.error
nb.lda.error.k5 <- nb.lda.error
nb.forward.error.k5 <- nb.forward.error
svm.acp.error.k5 <- svm.acp.error
svm.lda.error.k5 <- svm.lda.error
svm.forward.error.k5 <- svm.forward.error
svm.tune.acp.k5 <- svm.tune.acp
svm.tune.acp.error.k5 <- svm.tune.acp.error
svm.tune.lda.k5 <- svm.tune.lda
svm.tune.lda.error.k5 <- svm.tune.lda.error
svm.tune.forward.k5 <- svm.tune.forward
svm.tune.forward.error.k5 <- svm.tune.forward.error
tree.acp.error.k5 <- tree.acp.error
tree.lda.error.k5 <- tree.lda.error
tree.forward.error.k5 <- tree.forward.error
rf.acp.error.k5 <- rf.acp.error
rf.lda.error.k5 <- rf.lda.error
rf.forward.error.k5 <- rf.forward.error
nn.acp.error.k5 <- nn.acp.error
nn.lda.error.k5 <- nn.lda.error
nn.forward.error.k5 <- nn.forward.error

load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/end_10.rData")

vectorTree <- seq(100,1500,100)

# declaration de fonciton utile pour le reseau de neurones
getFormulas <- function(col, order, label) {
  result <- vector(mode="character", length=length(order))
  for(i in 1:length(order)){
    if(i == 1){
      result[i] <- paste(c(as.character(label)," ~ ",col[order[i]]),collapse = '')
    } else {
      result[i] <- paste(c(result[i-1], col[order[i]]), collapse = ' + ')
    }
  }
  return(result)
}

# ====
# Reduction de la dimensionnalite des variables
# ===
# 1) FDA
X.lda <- lda(y.app~., data=as.data.frame(X.app))
X.lda.data <- X.app%*%X.lda$scaling

# 2) ACP
X.acp <- prcomp(X.app)
X.acp.data <- X.acp$x[,1:160] 

# 3) Forward/Backward Selection
reg.fit<-regsubsets(y.app~.,data=as.data.frame(X.acp.data),method="forward", intercept=FALSE)

X.forward.data <- X.acp.data[,which(reg.fit$vorder == 1)]
X.forward.data <- as.data.frame(X.forward.data)
names(X.forward.data)[1] <- colnames(X.acp.data)[which(reg.fit$vorder == 1)]

y.test.app.factor <- factor(y.app)
y.test.test.factor <- factor(y.test)
levels(y.test.test.factor) <- levels(y.test.app.factor)


## === 
## Transformation aussi de nos donnees de test dans le meme referentiel pour toutes 
## ===

# 1) FDA
X.lda.test <- X.test%*%X.lda$scaling

# 2) ACP
X.acp.test <- X.test %*% X.acp$rotation[,1:160]

# 3) Forward
X.forward.test <- X.acp.test[,which(reg.fit$vorder == 1)]
X.forward.test <- as.data.frame(X.forward.test)
names(X.forward.test)[1] <- colnames(X.acp.test)[which(reg.fit$vorder == 1)]

for(j in 2:160){
  X.forward.data[[colnames(X.acp.data)[which(reg.fit$vorder == j)]]] <- X.acp.data[,which(reg.fit$vorder == j)]
  X.forward.test[[colnames(X.acp.test)[which(reg.fit$vorder == j)]]] <- X.acp.test[,which(reg.fit$vorder == j)]
}


# ===
# ACP
# ===
print("ACP")
# Random Forest
ind <- which(rf.acp.error == min(rf.acp.error), arr.ind = TRUE) 
numberofTree <- vectorTree[ind[1]]
rf.test.acp <- randomForest(y.test.app.factor~., data=as.data.frame(X.acp.data[,1:ind[2]]), xtest=as.data.frame(X.acp.test[,1:ind[2]]), ytest=y.test.test.factor, ntree=numberofTree)
rf.test.acp.error <- (1 - (sum(diag(rf.test.acp$test$confusion)/X.test.dim[1])))*100
rf.app.acp.error <- min(rf.acp.error)

ind <- which(rf.acp.error.k5 == min(rf.acp.error.k5), arr.ind = TRUE) 
numberofTree <- vectorTree[ind[1]]
rf.test.acp.k5 <- randomForest(y.test.app.factor~., data=as.data.frame(X.acp.data[,1:ind[2]]), xtest=as.data.frame(X.acp.test[,1:ind[2]]), ytest=y.test.test.factor, ntree=numberofTree)
rf.test.acp.error.k5 <- (1 - (sum(diag(rf.test.acp.k5$test$confusion)/X.test.dim[1])))*100
rf.app.acp.error.k5 <- min(rf.acp.error.k5)

# Tree
ind <- which.min(tree.acp.error)
tree.test.acp <- tree(factor(y.app)~., data=as.data.frame(X.acp.data[,1:ind]))
tree.test.acp.pred <- predict(tree.test.acp,newdata=as.data.frame(X.acp.test[,1:ind]), type="class")
tree.test.acp.perf <- table(y.test,tree.test.acp.pred)
tree.test.acp.error <- (1 - sum(diag(tree.test.acp.perf))/X.test.dim[1])*100
tree.app.acp.error <- min(tree.acp.error)

ind <- which.min(tree.acp.error.k5)
tree.test.acp.k5 <- tree(factor(y.app)~., data=as.data.frame(X.acp.data[,1:ind]))
tree.test.acp.pred.k5 <- predict(tree.test.acp.k5,newdata=as.data.frame(X.acp.test[,1:ind]), type="class")
tree.test.acp.perf.k5 <- table(y.test,tree.test.acp.pred.k5)
tree.test.acp.error.k5 <- (1 - sum(diag(tree.test.acp.perf.k5))/X.test.dim[1])*100
tree.app.acp.error.k5 <- min(tree.acp.error.k5)

# SVM
ind <- which.min(svm.acp.error)
svm.test.acp <- svm(X.acp.data[,1:ind],y.app,type="C-classification")
svm.test.acp.pred <- predict(svm.test.acp, X.acp.test[,1:ind])
svm.test.acp.perf <- table(factor(y.test),svm.test.acp.pred)
svm.test.acp.error <- (1 - sum(diag(svm.test.acp.perf))/X.test.dim[1])*100
svm.app.acp.error <- min(svm.acp.error)

ind <- which.min(svm.acp.error.k5)
svm.test.acp.k5 <- svm(X.acp.data[,1:ind],y.app,type="C-classification")
svm.test.acp.pred.k5 <- predict(svm.test.acp.k5, X.acp.test[,1:ind])
svm.test.acp.perf.k5 <- table(factor(y.test),svm.test.acp.pred.k5)
svm.test.acp.error.k5 <- (1 - sum(diag(svm.test.acp.perf.k5))/X.test.dim[1])*100
svm.app.acp.error.k5 <- min(svm.acp.error.k5)

# SVM Tune
ind <- which.min(svm.tune.acp.error)
svm.tune.test.acp <- tune(svm, train.y = factor(y.app),  train.x = X.acp.data[,1:ind], cost=svm.tune.acp$best.parameters$cost, gamma=svm.tune.acp$best.parameters$gamma)
svm.tune.test.acp.pred <- predict(svm.tune.test.acp$best.model , X.acp.test[,1:ind])
svm.tune.test.acp.perf <- table(factor(y.test),svm.tune.test.acp.pred)
svm.tune.test.acp.error <- (1 - sum(diag(svm.tune.test.acp.perf))/X.test.dim[1])*100
svm.tune.app.acp.error <- min(svm.tune.acp.error)

ind <- which.min(svm.tune.acp.error.k5)
svm.tune.test.acp.k5 <- tune(svm, train.y = factor(y.app),  train.x = X.acp.data[,1:ind], cost=svm.tune.acp.k5$best.parameters$cost, gamma=svm.tune.acp.k5$best.parameters$gamma)
svm.tune.test.acp.pred.k5 <- predict(svm.tune.test.acp.k5$best.model , X.acp.test[,1:ind])
svm.tune.test.acp.perf.k5 <- table(factor(y.test),svm.tune.test.acp.pred.k5)
svm.tune.test.acp.error.k5 <- (1 - sum(diag(svm.tune.test.acp.perf.k5))/X.test.dim[1])*100
svm.tune.app.acp.error.k5 <- min(svm.tune.acp.error.k5)

# Naive Bayesien
ind <- which.min(nb.acp.error)
nb.test.acp <- naiveBayes(factor(y.app)~., data=as.data.frame(X.acp.data[,1:ind]))
nb.test.acp.pred <- predict(nb.test.acp,newdata=as.data.frame(X.acp.test[,1:ind]))
nb.test.acp.perf <- table(factor(y.test),nb.test.acp.pred)
nb.test.acp.error <- (1 - sum(diag(nb.test.acp.perf))/X.test.dim[1])*100
nb.app.acp.error <- min(nb.acp.error)

ind <- which.min(nb.acp.error.k5)
nb.test.acp.k5 <- naiveBayes(factor(y.app)~., data=as.data.frame(X.acp.data[,1:ind]))
nb.test.acp.pred.k5 <- predict(nb.test.acp.k5,newdata=as.data.frame(X.acp.test[,1:ind]))
nb.test.acp.perf.k5 <- table(factor(y.test),nb.test.acp.pred.k5)
nb.test.acp.error.k5 <- (1 - sum(diag(nb.test.acp.perf.k5))/X.test.dim[1])*100
nb.app.acp.error.k5 <- min(nb.acp.error.k5)

# RegLog
ind <- which.min(logReg.acp.error)
logReg.test.acp.res<-c(rep(0,X.test.dim[1]))
logReg.test.acp <- glmnet(X.acp.data[,1:ind] ,y=y.app,family="multinomial")
logReg.test.acp.pred <- predict(logReg.test.acp,newx=X.acp.test[,1:ind],type="response",s=logReg.test.acp$lambda.min)

for (h in 1:X.test.dim[1]) {
  logReg.test.acp.res[h] <-which.max(logReg.test.acp.pred[h,1:6,dim(logReg.test.acp.pred)[3]-1])
}
logReg.test.acp.perf <- table(y.test,logReg.test.acp.res)
logReg.test.acp.error <-(1 - sum(diag(logReg.test.acp.perf))/X.test.dim[1])*100
logReg.app.acp.error <- min(logReg.acp.error)

ind <- which.min(logReg.acp.error.k5)
logReg.test.acp.res.k5<-c(rep(0,X.test.dim[1]))
logReg.test.acp.k5 <- glmnet(X.acp.data[,1:ind] ,y=y.app,family="multinomial")
logReg.test.acp.pred.k5 <- predict(logReg.test.acp.k5,newx=X.acp.test[,1:ind],type="response",s=logReg.test.acp.k5$lambda.min)

for (h in 1:X.test.dim[1]) {
  logReg.test.acp.res.k5[h] <-which.max(logReg.test.acp.pred.k5[h,1:6,dim(logReg.test.acp.pred.k5)[3]-1])
}
logReg.test.acp.perf.k5 <- table(y.test,logReg.test.acp.res.k5)
logReg.test.acp.error.k5 <-(1 - sum(diag(logReg.test.acp.perf.k5))/X.test.dim[1])*100
logReg.app.acp.error.k5 <- min(logReg.acp.error.k5)

# LDA
ind <- which.min(lda.acp.error)
lda.test.acp <- lda(y.app~., data=as.data.frame(X.acp.data[,1:ind]))
lda.test.acp.pred <- predict(lda.test.acp,newdata=as.data.frame(X.acp.test[,1:ind]))
lda.test.acp.perf <- table(y.test,lda.test.acp.pred$class)
lda.test.acp.error <- (1 - sum(diag(lda.test.acp.perf))/X.test.dim[1])*100
lda.app.acp.error <- min(lda.acp.error)

ind <- which.min(lda.acp.error.k5)
lda.test.acp.k5 <- lda(y.app~., data=as.data.frame(X.acp.data[,1:ind]))
lda.test.acp.pred.k5 <- predict(lda.test.acp.k5,newdata=as.data.frame(X.acp.test[,1:ind]))
lda.test.acp.perf.k5 <- table(y.test,lda.test.acp.pred.k5$class)
lda.test.acp.error.k5 <- (1 - sum(diag(lda.test.acp.perf.k5))/X.test.dim[1])*100
lda.app.acp.error.k5 <- min(lda.acp.error.k5)

# KNN
ind <-   which(knn.acp.error == min(knn.acp.error), arr.ind = TRUE) 
knn.test.acp <- knn(as.data.frame(X.acp.data[,1:ind[2]]), as.data.frame(X.acp.test[,1:ind[2]]), y.app,k=ind)
knn.test.acp.perf <- table(y.test, knn.test.acp)
knn.test.acp.error <- (1 - sum(diag(knn.test.acp.perf))/X.test.dim[1])*100
knn.app.acp.error <- min(knn.acp.error)

ind <-  which(knn.acp.error.k5 == min(knn.acp.error.k5), arr.ind = TRUE) 
knn.test.acp.k5 <- knn(as.data.frame(X.acp.data[,1:ind[2]]), as.data.frame(X.acp.test[,1:ind[2]]), y.app,k=ind)
knn.test.acp.perf.k5 <- table(y.test, knn.test.acp.k5)
knn.test.acp.error.k5 <- (1 - sum(diag(knn.test.acp.perf.k5))/X.test.dim[1])*100
knn.app.acp.error.k5 <- min(knn.acp.error.k5)

# Neural Network
ind <- which.min(nn.acp.error)
data.train = X.acp.data[,1:ind]
data.test = X.acp.test[,1:ind]
y.train = y.app
y.testfold = y.test
ordre = c(1:dim(data.train)[2])
for(kk in 1:6){
  response <- rep(0,length(y.train))
  for(myVar in 1:length(y.train)){
    if(y.train[myVar]==kk){
      response[myVar]=1
    }
  }
  newDataSet <- data.frame(data.train,response)
  formule = getFormulas(colnames(newDataSet), ordre,"response")
  if(kk==1){
    neuralnet1 <- NULL
    while(is.null(neuralnet1$result.matrix)){
      neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if( kk==2) {
    neuralnet2 <- NULL
    while(is.null(neuralnet2$result.matrix)){
      neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==3){
    neuralnet3 <- NULL
    while(is.null(neuralnet3$result.matrix)){
      neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==4){
    neuralnet4 <- NULL
    while(is.null(neuralnet4$result.matrix)){
      neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==5){
    neuralnet5 <- NULL
    while(is.null(neuralnet5$result.matrix)){
      neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==6){
    neuralnet6 <- NULL
    while(is.null(neuralnet6$result.matrix)){
      neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  }
}
# predictions 
c1<-compute(neuralnet1,as.matrix(data.test))
c2<-compute(neuralnet2,as.matrix(data.test))
c3<-compute(neuralnet3,as.matrix(data.test))
c4<-compute(neuralnet4,as.matrix(data.test))
c5<-compute(neuralnet5,as.matrix(data.test))
c6<-compute(neuralnet6,as.matrix(data.test))
neuralnet.result <- rep(0,dim(data.test)[1])
for(myVar in 1:dim(data.test)[1]){
  index <- which.max(c(c1$net.result[myVar],c2$net.result[myVar],c3$net.result[myVar],c4$net.result[myVar],c5$net.result[myVar],c6$net.result[myVar]))
  neuralnet.result[myVar] <- index
}
neuralnet.perf <- table(neuralnet.result,y.testfold)
nn.test.acp.error <- (1-sum(diag(neuralnet.perf))/length(y.testfold))*100
nn.app.acp.error <- min(nn.acp.error)

ind <- which.min(nn.acp.error.k5)
data.train = X.acp.data[,1:ind]
data.test = X.acp.test[,1:ind]
y.train = y.app
y.testfold = y.test
ordre = c(1:dim(data.train)[2])
for(kk in 1:6){
  response <- rep(0,length(y.train))
  for(myVar in 1:length(y.train)){
    if(y.train[myVar]==kk){
      response[myVar]=1
    }
  }
  newDataSet <- data.frame(data.train,response)
  formule = getFormulas(colnames(newDataSet), ordre,"response")
  if(kk==1){
    neuralnet1.k5 <- NULL
    while(is.null(neuralnet1.k5$result.matrix)){
      neuralnet1.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if( kk==2) {
    neuralnet2.k5 <- NULL
    while(is.null(neuralnet2.k5$result.matrix)){
      neuralnet2.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==3){
    neuralnet3.k5 <- NULL
    while(is.null(neuralnet3.k5$result.matrix)){
      neuralnet3.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==4){
    neuralnet4.k5 <- NULL
    while(is.null(neuralnet4.k5$result.matrix)){
      neuralnet4.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==5){
    neuralnet5.k5 <- NULL
    while(is.null(neuralnet5.k5$result.matrix)){
      neuralnet5.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==6){
    neuralnet6.k5 <- NULL
    while(is.null(neuralnet6.k5$result.matrix)){
      neuralnet6.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  }
}
# predictions 
c1<-compute(neuralnet1.k5,as.matrix(data.test))
c2<-compute(neuralnet2.k5,as.matrix(data.test))
c3<-compute(neuralnet3.k5,as.matrix(data.test))
c4<-compute(neuralnet4.k5,as.matrix(data.test))
c5<-compute(neuralnet5.k5,as.matrix(data.test))
c6<-compute(neuralnet6.k5,as.matrix(data.test))
neuralnet.result.k5 <- rep(0,dim(data.test)[1])
for(myVar in 1:dim(data.test)[1]){
  index <- which.max(c(c1$net.result[myVar],c2$net.result[myVar],c3$net.result[myVar],c4$net.result[myVar],c5$net.result[myVar],c6$net.result[myVar]))
  neuralnet.result.k5[myVar] <- index
}
neuralnet.perf.k5 <- table(neuralnet.result.k5,y.testfold)
nn.test.acp.error.k5 <- (1-sum(diag(neuralnet.perf.k5))/length(y.testfold))*100
nn.app.acp.error.k5 <- min(nn.acp.error.k5)

# ===
# Forward
# ===
print("Forward")
ind <-  which(rf.forward.error == min(rf.forward.error), arr.ind = TRUE) 
numberofTree <- vectorTree[ind[1]]
rf.test.forward <- randomForest(y.test.app.factor~., data=as.data.frame(X.forward.data[,1:ind[2]]), xtest=as.data.frame(X.forward.test[,1:ind[2]]), ytest=y.test.test.factor, ntree=numberofTree)
rf.test.forward.error <- (1 - (sum(diag(rf.test.forward$test$confusion)/X.test.dim[1])))*100
rf.app.forward.error <- min(rf.forward.error)

ind <-  which(rf.forward.error.k5 == min(rf.forward.error.k5), arr.ind = TRUE)
numberofTree <- vectorTree[ind[1]]
rf.test.forward.k5 <- randomForest(y.test.app.factor~., data=as.data.frame(X.forward.data[,1:ind[2]]), xtest=as.data.frame(X.forward.test[,1:ind[2]]), ytest=y.test.test.factor, ntree=numberofTree)
rf.test.forward.error.k5 <- (1 - (sum(diag(rf.test.forward.k5$test$confusion)/X.test.dim[1])))*100
rf.app.forward.error.k5 <- min(rf.forward.error.k5)

# Tree
ind <- which.min(tree.forward.error)
tree.test.forward <- tree(factor(y.app)~., data=as.data.frame(X.forward.data[,1:ind]))
tree.test.forward.pred <- predict(tree.test.forward,newdata=as.data.frame(X.forward.test[,1:ind]), type="class")
tree.test.forward.perf <- table(y.test,tree.test.forward.pred)
tree.test.forward.error <- (1 - sum(diag(tree.test.forward.perf))/X.test.dim[1])*100
tree.app.forward.error <- min(tree.forward.error)

ind <- which.min(tree.forward.error.k5)
tree.test.forward.k5 <- tree(factor(y.app)~., data=as.data.frame(X.forward.data[,1:ind]))
tree.test.forward.pred.k5 <- predict(tree.test.forward.k5,newdata=as.data.frame(X.forward.test[,1:ind]), type="class")
tree.test.forward.perf.k5 <- table(y.test,tree.test.forward.pred.k5)
tree.test.forward.error.k5 <- (1 - sum(diag(tree.test.forward.perf.k5))/X.test.dim[1])*100
tree.app.forward.error.k5 <- min(tree.forward.error.k5)

# SVM
ind <- which.min(svm.forward.error)
svm.test.forward <- svm(X.forward.data[,1:ind],y.app,type="C-classification")
svm.test.forward.pred <- predict(svm.test.forward, X.forward.test[,1:ind])
svm.test.forward.perf <- table(factor(y.test),svm.test.forward.pred)
svm.test.forward.error <- (1 - sum(diag(svm.test.forward.perf))/X.test.dim[1])*100
svm.app.forward.error <- min(svm.forward.error)

ind <- which.min(svm.forward.error.k5)
svm.test.forward.k5 <- svm(X.forward.data[,1:ind],y.app,type="C-classification")
svm.test.forward.pred.k5 <- predict(svm.test.forward.k5, X.forward.test[,1:ind])
svm.test.forward.perf.k5 <- table(factor(y.test),svm.test.forward.pred.k5)
svm.test.forward.error.k5 <- (1 - sum(diag(svm.test.forward.perf.k5))/X.test.dim[1])*100
svm.app.forward.error.k5 <- min(svm.forward.error.k5)

# SVM Tune
ind <- which.min(svm.tune.forward.error)
svm.tune.test.forward <- tune(svm, train.y = factor(y.app),  train.x = X.forward.data[,1:ind], cost=svm.tune.forward$best.parameters$cost, gamma=svm.tune.forward$best.parameters$gamma)
svm.tune.test.forward.pred <- predict(svm.tune.test.forward$best.model , X.forward.test[,1:ind])
svm.tune.test.forward.perf <- table(factor(y.test),svm.tune.test.forward.pred)
svm.tune.test.forward.error <- (1 - sum(diag(svm.tune.test.forward.perf))/X.test.dim[1])*100
svm.tune.app.forward.error <- min(svm.tune.forward.error)

ind <- which.min(svm.tune.forward.error.k5)
svm.tune.test.forward.k5 <- tune(svm, train.y = factor(y.app),  train.x = X.forward.data[,1:ind], cost=svm.tune.forward.k5$best.parameters$cost, gamma=svm.tune.forward.k5$best.parameters$gamma)
svm.tune.test.forward.pred.k5 <- predict(svm.tune.test.forward.k5$best.model , X.forward.test[,1:ind])
svm.tune.test.forward.perf.k5 <- table(factor(y.test),svm.tune.test.forward.pred.k5)
svm.tune.test.forward.error.k5 <- (1 - sum(diag(svm.tune.test.forward.perf.k5))/X.test.dim[1])*100
svm.tune.app.forward.error.k5 <- min(svm.tune.forward.error.k5)

# Naive Bayesien
ind <- which.min(nb.forward.error)
nb.test.forward <- naiveBayes(factor(y.app)~., data=as.data.frame(X.forward.data[,1:ind]))
nb.test.forward.pred <- predict(nb.test.forward,newdata=as.data.frame(X.forward.test[,1:ind]))
nb.test.forward.perf <- table(factor(y.test),nb.test.forward.pred)
nb.test.forward.error <- (1 - sum(diag(nb.test.forward.perf))/X.test.dim[1])*100
nb.app.forward.error <- min(nb.forward.error)

ind <- which.min(nb.forward.error.k5)
nb.test.forward.k5 <- naiveBayes(factor(y.app)~., data=as.data.frame(X.forward.data[,1:ind]))
nb.test.forward.pred.k5 <- predict(nb.test.forward.k5,newdata=as.data.frame(X.forward.test[,1:ind]))
nb.test.forward.perf.k5 <- table(factor(y.test),nb.test.forward.pred.k5)
nb.test.forward.error.k5 <- (1 - sum(diag(nb.test.forward.perf.k5))/X.test.dim[1])*100
nb.app.forward.error.k5 <- min(nb.forward.error.k5)

# RegLog
ind <- which.min(logReg.forward.error)
logReg.test.forward.res<-c(rep(0,X.test.dim[1]))
logReg.test.forward <- glmnet(as.matrix(X.forward.data[,1:ind]) ,y=y.app,family="multinomial")
logReg.test.forward.pred <- predict(logReg.test.forward,newx=as.matrix(X.forward.test[,1:ind]),type="response",s=logReg.test.forward$lambda.min)

for (h in 1:X.test.dim[1]) {
  logReg.test.forward.res[h] <-which.max(logReg.test.forward.pred[h,1:6,dim(logReg.test.forward.pred)[3]-1])
}
logReg.test.forward.perf <- table(y.test,logReg.test.forward.res)
logReg.test.forward.error <-(1 - sum(diag(logReg.test.forward.perf))/X.test.dim[1])*100
logReg.app.forward.error <- min(logReg.forward.error)

ind <- which.min(logReg.forward.error.k5)
logReg.test.forward.res.k5<-c(rep(0,X.test.dim[1]))
logReg.test.forward.k5 <- glmnet(as.matrix(X.forward.data[,1:ind]) ,y=y.app,family="multinomial")
logReg.test.forward.pred.k5 <- predict(logReg.test.forward.k5,newx=as.matrix(X.forward.test[,1:ind]),type="response",s=logReg.test.forward.k5$lambda.min)

for (h in 1:X.test.dim[1]) {
  logReg.test.forward.res.k5[h] <-which.max(logReg.test.forward.pred.k5[h,1:6,dim(logReg.test.forward.pred.k5)[3]-1])
}
logReg.test.forward.perf.k5 <- table(y.test,logReg.test.forward.res.k5)
logReg.test.forward.error.k5 <-(1 - sum(diag(logReg.test.forward.perf.k5))/X.test.dim[1])*100
logReg.app.forward.error.k5 <- min(logReg.forward.error.k5)

# LDA
ind <- which.min(lda.forward.error)
lda.test.forward <- lda(y.app~., data=as.data.frame(X.forward.data[,1:ind]))
lda.test.forward.pred <- predict(lda.test.forward,newdata=as.data.frame(X.forward.test[,1:ind]))
lda.test.forward.perf <- table(y.test,lda.test.forward.pred$class)
lda.test.forward.error <- (1 - sum(diag(lda.test.forward.perf))/X.test.dim[1])*100
lda.app.forward.error <- min(lda.forward.error)

ind <- which.min(lda.forward.error.k5)
lda.test.forward.k5 <- lda(y.app~., data=as.data.frame(X.forward.data[,1:ind]))
lda.test.forward.pred.k5 <- predict(lda.test.forward.k5,newdata=as.data.frame(X.forward.test[,1:ind]))
lda.test.forward.perf.k5 <- table(y.test,lda.test.forward.pred.k5$class)
lda.test.forward.error.k5 <- (1 - sum(diag(lda.test.forward.perf.k5))/X.test.dim[1])*100
lda.app.forward.error.k5 <- min(lda.forward.error.k5)

# KNN
ind <-  which(knn.forward.error == min(knn.forward.error), arr.ind = TRUE) 
knn.test.forward <- knn(as.data.frame(X.forward.data[,1:ind[2]]), as.data.frame(X.forward.test[,1:ind[2]]), y.app,k=ind)
knn.test.forward.perf <- table(y.test, knn.test.forward)
knn.test.forward.error <- (1 - sum(diag(knn.test.forward.perf))/X.test.dim[1])*100
knn.app.forward.error <- min(knn.forward.error)

ind <-  which(knn.forward.error.k5 == min(knn.forward.error.k5), arr.ind = TRUE) 
knn.test.forward.k5 <- knn(as.data.frame(X.forward.data[,1:ind[2]]), as.data.frame(X.forward.test[,1:ind[2]]), y.app,k=ind)
knn.test.forward.perf.k5 <- table(y.test, knn.test.forward.k5)
knn.test.forward.error.k5 <- (1 - sum(diag(knn.test.forward.perf.k5))/X.test.dim[1])*100
knn.app.forward.error.k5 <- min(knn.forward.error.k5)

# Neural Network
ind <- which.min(nn.forward.error)
data.train = X.forward.data[,1:ind]
data.test = X.forward.test[,1:ind]
y.train = y.app
y.testfold = y.test
ordre = c(1:dim(data.train)[2])
for(kk in 1:6){
  response <- rep(0,length(y.train))
  for(myVar in 1:length(y.train)){
    if(y.train[myVar]==kk){
      response[myVar]=1
    }
  }
  newDataSet <- data.frame(data.train,response)
  formule = getFormulas(colnames(newDataSet), ordre,"response")
  if(kk==1){
    neuralnet1 <- NULL
    while(is.null(neuralnet1$result.matrix)){
      neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if( kk==2) {
    neuralnet2 <- NULL
    while(is.null(neuralnet2$result.matrix)){
      neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==3){
    neuralnet3 <- NULL
    while(is.null(neuralnet3$result.matrix)){
      neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==4){
    neuralnet4 <- NULL
    while(is.null(neuralnet4$result.matrix)){
      neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==5){
    neuralnet5 <- NULL
    while(is.null(neuralnet5$result.matrix)){
      neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==6){
    neuralnet6 <- NULL
    while(is.null(neuralnet6$result.matrix)){
      neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  }
}
# predictions 
c1<-compute(neuralnet1,as.matrix(data.test))
c2<-compute(neuralnet2,as.matrix(data.test))
c3<-compute(neuralnet3,as.matrix(data.test))
c4<-compute(neuralnet4,as.matrix(data.test))
c5<-compute(neuralnet5,as.matrix(data.test))
c6<-compute(neuralnet6,as.matrix(data.test))
neuralnet.result <- rep(0,dim(data.test)[1])
for(myVar in 1:dim(data.test)[1]){
  index <- which.max(c(c1$net.result[myVar],c2$net.result[myVar],c3$net.result[myVar],c4$net.result[myVar],c5$net.result[myVar],c6$net.result[myVar]))
  neuralnet.result[myVar] <- index
}
neuralnet.perf <- table(neuralnet.result,y.testfold)
nn.test.forward.error <- (1-sum(diag(neuralnet.perf))/length(y.testfold))*100
nn.app.forward.error <- min(nn.forward.error)

ind <- which.min(nn.forward.error.k5)
data.train = X.forward.data[,1:ind]
data.test = X.forward.test[,1:ind]
y.train = y.app
y.testfold = y.test
ordre = c(1:dim(data.train)[2])
for(kk in 1:6){
  response <- rep(0,length(y.train))
  for(myVar in 1:length(y.train)){
    if(y.train[myVar]==kk){
      response[myVar]=1
    }
  }
  newDataSet <- data.frame(data.train,response)
  formule = getFormulas(colnames(newDataSet), ordre,"response")
  if(kk==1){
    neuralnet1.k5 <- NULL
    while(is.null(neuralnet1.k5$result.matrix)){
      neuralnet1.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if( kk==2) {
    neuralnet2.k5 <- NULL
    while(is.null(neuralnet2.k5$result.matrix)){
      neuralnet2.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==3){
    neuralnet3.k5 <- NULL
    while(is.null(neuralnet3.k5$result.matrix)){
      neuralnet3.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==4){
    neuralnet4.k5 <- NULL
    while(is.null(neuralnet4.k5$result.matrix)){
      neuralnet4.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==5){
    neuralnet5.k5 <- NULL
    while(is.null(neuralnet5.k5$result.matrix)){
      neuralnet5.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==6){
    neuralnet6.k5 <- NULL
    while(is.null(neuralnet6.k5$result.matrix)){
      neuralnet6.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  }
}
# predictions 
c1<-compute(neuralnet1.k5,as.matrix(data.test))
c2<-compute(neuralnet2.k5,as.matrix(data.test))
c3<-compute(neuralnet3.k5,as.matrix(data.test))
c4<-compute(neuralnet4.k5,as.matrix(data.test))
c5<-compute(neuralnet5.k5,as.matrix(data.test))
c6<-compute(neuralnet6.k5,as.matrix(data.test))
neuralnet.result.k5 <- rep(0,dim(data.test)[1])
for(myVar in 1:dim(data.test)[1]){
  index <- which.max(c(c1$net.result[myVar],c2$net.result[myVar],c3$net.result[myVar],c4$net.result[myVar],c5$net.result[myVar],c6$net.result[myVar]))
  neuralnet.result.k5[myVar] <- index
}
neuralnet.perf.k5 <- table(neuralnet.result.k5,y.testfold)
nn.test.forward.error.k5 <- (1-sum(diag(neuralnet.perf.k5))/length(y.testfold))*100
nn.app.forward.error.k5 <- min(nn.forward.error.k5)


# ===
# FDA
# ===
print("FDA")
# Random Forest
ind <-  which.min(rf.lda.error)
numberofTree <- vectorTree[ind]
rf.test.lda <- randomForest(y.test.app.factor~., data=as.data.frame(X.lda.data), xtest=as.data.frame(X.lda.test), ytest=y.test.test.factor, ntree=numberofTree)
rf.test.lda.error <- (1 - (sum(diag(rf.test.lda$test$confusion)/X.test.dim[1])))*100
rf.app.lda.error <- min(rf.lda.error)

ind <-  which.min(rf.lda.error.k5)
numberofTree <- vectorTree[ind]
rf.test.lda.k5 <- randomForest(y.test.app.factor~., data=as.data.frame(X.lda.data), xtest=as.data.frame(X.lda.test), ytest=y.test.test.factor, ntree=numberofTree)
rf.test.lda.error.k5 <- (1 - (sum(diag(rf.test.lda.k5$test$confusion)/X.test.dim[1])))*100
rf.app.lda.error.k5 <- min(rf.lda.error.k5)

# Tree
tree.test.lda <- tree(factor(y.app)~., data=as.data.frame(X.lda.data))
tree.test.lda.pred <- predict(tree.test.lda,newdata=as.data.frame(X.lda.test), type="class")
tree.test.lda.perf <- table(y.test,tree.test.lda.pred)
tree.test.lda.error <- (1 - sum(diag(tree.test.lda.perf))/X.test.dim[1])*100
tree.app.lda.error <- min(tree.lda.error)

tree.test.lda.k5 <- tree(factor(y.app)~., data=as.data.frame(X.lda.data))
tree.test.lda.pred.k5 <- predict(tree.test.lda.k5,newdata=as.data.frame(X.lda.test), type="class")
tree.test.lda.perf.k5 <- table(y.test,tree.test.lda.pred.k5)
tree.test.lda.error.k5 <- (1 - sum(diag(tree.test.lda.perf.k5))/X.test.dim[1])*100
tree.app.lda.error.k5 <- min(tree.lda.error.k5)

# SVM
svm.test.lda <- svm(X.lda.data,y.app,type="C-classification")
svm.test.lda.pred <- predict(svm.test.lda, X.lda.test)
svm.test.lda.perf <- table(factor(y.test),svm.test.lda.pred)
svm.test.lda.error <- (1 - sum(diag(svm.test.lda.perf))/X.test.dim[1])*100
svm.app.lda.error <- min(svm.lda.error)

svm.test.lda.k5 <- svm(X.lda.data,y.app,type="C-classification")
svm.test.lda.pred.k5 <- predict(svm.test.lda.k5, X.lda.test)
svm.test.lda.perf.k5 <- table(factor(y.test),svm.test.lda.pred.k5)
svm.test.lda.error.k5 <- (1 - sum(diag(svm.test.lda.perf.k5))/X.test.dim[1])*100
svm.app.lda.error.k5 <- min(svm.lda.error.k5)

# SVM Tune
svm.tune.test.lda <- tune(svm, train.y = factor(y.app),  train.x = X.lda.data, cost=svm.tune.lda$best.parameters$cost, gamma=svm.tune.lda$best.parameters$gamma)
svm.tune.test.lda.pred <- predict(svm.tune.test.lda$best.model , X.lda.test)
svm.tune.test.lda.perf <- table(factor(y.test),svm.tune.test.lda.pred)
svm.tune.test.lda.error <- (1 - sum(diag(svm.tune.test.lda.perf))/X.test.dim[1])*100
svm.tune.app.lda.error <- min(svm.tune.lda.error)

svm.tune.test.lda.k5 <- tune(svm, train.y = factor(y.app),  train.x = X.lda.data, cost=svm.tune.lda.k5$best.parameters$cost, gamma=svm.tune.lda.k5$best.parameters$gamma)
svm.tune.test.lda.pred.k5 <- predict(svm.tune.test.lda.k5$best.model , X.lda.test)
svm.tune.test.lda.perf.k5 <- table(factor(y.test),svm.tune.test.lda.pred.k5)
svm.tune.test.lda.error.k5 <- (1 - sum(diag(svm.tune.test.lda.perf.k5))/X.test.dim[1])*100
svm.tune.app.lda.error.k5 <- min(svm.tune.lda.error.k5)

# Naive Bayesien
nb.test.lda <- naiveBayes(factor(y.app)~., data=as.data.frame(X.lda.data))
nb.test.lda.pred <- predict(nb.test.lda,newdata=as.data.frame(X.lda.test))
nb.test.lda.perf <- table(factor(y.test),nb.test.lda.pred)
nb.test.lda.error <- (1 - sum(diag(nb.test.lda.perf))/X.test.dim[1])*100
nb.app.lda.error <- min(nb.lda.error)

nb.test.lda.k5 <- naiveBayes(factor(y.app)~., data=as.data.frame(X.lda.data))
nb.test.lda.pred.k5 <- predict(nb.test.lda.k5,newdata=as.data.frame(X.lda.test))
nb.test.lda.perf.k5 <- table(factor(y.test),nb.test.lda.pred.k5)
nb.test.lda.error.k5 <- (1 - sum(diag(nb.test.lda.perf.k5))/X.test.dim[1])*100
nb.app.lda.error.k5 <- min(nb.lda.error.k5)

# RegLog
logReg.test.lda.res<-c(rep(0,X.test.dim[1]))
logReg.test.lda <- glmnet(X.lda.data ,y=y.app,family="multinomial")
logReg.test.lda.pred <- predict(logReg.test.lda,newx=X.lda.test,type="response",s=logReg.test.lda$lambda.min)

for (h in 1:X.test.dim[1]) {
  logReg.test.lda.res[h] <-which.max(logReg.test.lda.pred[h,1:6,dim(logReg.test.lda.pred)[3]-1])
}
logReg.test.lda.perf <- table(y.test,logReg.test.lda.res)
logReg.test.lda.error <-(1 - sum(diag(logReg.test.lda.perf))/X.test.dim[1])*100
logReg.app.lda.error <- min(logReg.lda.error)

logReg.test.lda.res.k5<-c(rep(0,X.test.dim[1]))
logReg.test.lda.k5 <- glmnet(X.lda.data ,y=y.app,family="multinomial")
logReg.test.lda.pred.k5 <- predict(logReg.test.lda.k5,newx=X.lda.test,type="response",s=logReg.test.lda.k5$lambda.min)

for (h in 1:X.test.dim[1]) {
  logReg.test.lda.res.k5[h] <-which.max(logReg.test.lda.pred.k5[h,1:6,dim(logReg.test.lda.pred.k5)[3]-1])
}
logReg.test.lda.perf.k5 <- table(y.test,logReg.test.lda.res.k5)
logReg.test.lda.error.k5 <-(1 - sum(diag(logReg.test.lda.perf.k5))/X.test.dim[1])*100
logReg.app.lda.error.k5 <- min(logReg.lda.error.k5)

# LDA
lda.test.lda <- lda(y.app~., data=as.data.frame(X.lda.data))
lda.test.lda.pred <- predict(lda.test.lda,newdata=as.data.frame(X.lda.test))
lda.test.lda.perf <- table(y.test,lda.test.lda.pred$class)
lda.test.lda.error <- (1 - sum(diag(lda.test.lda.perf))/X.test.dim[1])*100
lda.app.lda.error <- min(lda.lda.error)

lda.test.lda.k5 <- lda(y.app~., data=as.data.frame(X.lda.data))
lda.test.lda.pred.k5 <- predict(lda.test.lda.k5,newdata=as.data.frame(X.lda.test))
lda.test.lda.perf.k5 <- table(y.test,lda.test.lda.pred.k5$class)
lda.test.lda.error.k5 <- (1 - sum(diag(lda.test.lda.perf.k5))/X.test.dim[1])*100
lda.app.lda.error.k5 <- min(lda.lda.error.k5)

# KNN
ind <-  which.min(knn.lda.error)
knn.test.lda <- knn(as.data.frame(X.lda.data), as.data.frame(X.lda.test), y.app,k=ind)
knn.test.lda.perf <- table(y.test, knn.test.lda)
knn.test.lda.error <- (1 - sum(diag(knn.test.lda.perf))/X.test.dim[1])*100
knn.app.lda.error <- min(knn.lda.error)

ind <-  which.min(knn.lda.error.k5)
knn.test.lda.k5 <- knn(as.data.frame(X.lda.data), as.data.frame(X.lda.test), y.app,k=ind)
knn.test.lda.perf.k5 <- table(y.test, knn.test.lda.k5)
knn.test.lda.error.k5 <- (1 - sum(diag(knn.test.lda.perf.k5))/X.test.dim[1])*100
knn.app.lda.error.k5 <- min(knn.lda.error.k5)

# Neural Network
data.train = X.lda.data
data.test = X.lda.test
y.train = y.app
y.testfold = y.test
ordre = c(1:dim(data.train)[2])
for(kk in 1:6){
  response <- rep(0,length(y.train))
  for(myVar in 1:length(y.train)){
    if(y.train[myVar]==kk){
      response[myVar]=1
    }
  }
  newDataSet <- data.frame(data.train,response)
  formule = getFormulas(colnames(newDataSet), ordre,"response")
  if(kk==1){
    neuralnet1 <- NULL
    while(is.null(neuralnet1$result.matrix)){
      neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if( kk==2) {
    neuralnet2 <- NULL
    while(is.null(neuralnet2$result.matrix)){
      neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==3){
    neuralnet3 <- NULL
    while(is.null(neuralnet3$result.matrix)){
      neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==4){
    neuralnet4 <- NULL
    while(is.null(neuralnet4$result.matrix)){
      neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==5){
    neuralnet5 <- NULL
    while(is.null(neuralnet5$result.matrix)){
      neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==6){
    neuralnet6 <- NULL
    while(is.null(neuralnet6$result.matrix)){
      neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  }
}
# predictions 
c1<-compute(neuralnet1,as.matrix(data.test))
c2<-compute(neuralnet2,as.matrix(data.test))
c3<-compute(neuralnet3,as.matrix(data.test))
c4<-compute(neuralnet4,as.matrix(data.test))
c5<-compute(neuralnet5,as.matrix(data.test))
c6<-compute(neuralnet6,as.matrix(data.test))
neuralnet.result <- rep(0,dim(data.test)[1])
for(myVar in 1:dim(data.test)[1]){
  index <- which.max(c(c1$net.result[myVar],c2$net.result[myVar],c3$net.result[myVar],c4$net.result[myVar],c5$net.result[myVar],c6$net.result[myVar]))
  neuralnet.result[myVar] <- index
}
neuralnet.perf <- table(neuralnet.result,y.testfold)
nn.test.lda.error <- (1-sum(diag(neuralnet.perf))/length(y.testfold))*100
nn.app.lda.error <- min(nn.lda.error)


data.train = X.lda.data
data.test = X.lda.test
y.train = y.app
y.testfold = y.test
ordre = c(1:dim(data.train)[2])
for(kk in 1:6){
  response <- rep(0,length(y.train))
  for(myVar in 1:length(y.train)){
    if(y.train[myVar]==kk){
      response[myVar]=1
    }
  }
  newDataSet <- data.frame(data.train,response)
  formule = getFormulas(colnames(newDataSet), ordre,"response")
  if(kk==1){
    neuralnet1.k5 <- NULL
    while(is.null(neuralnet1.k5$result.matrix)){
      neuralnet1.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if( kk==2) {
    neuralnet2.k5 <- NULL
    while(is.null(neuralnet2.k5$result.matrix)){
      neuralnet2.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==3){
    neuralnet3.k5 <- NULL
    while(is.null(neuralnet3.k5$result.matrix)){
      neuralnet3.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==4){
    neuralnet4.k5 <- NULL
    while(is.null(neuralnet4.k5$result.matrix)){
      neuralnet4.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==5){
    neuralnet5.k5 <- NULL
    while(is.null(neuralnet5.k5$result.matrix)){
      neuralnet5.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  } else if(kk==6){
    neuralnet6.k5 <- NULL
    while(is.null(neuralnet6.k5$result.matrix)){
      neuralnet6.k5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
    }
  }
}
# predictions 
c1<-compute(neuralnet1.k5,as.matrix(data.test))
c2<-compute(neuralnet2.k5,as.matrix(data.test))
c3<-compute(neuralnet3.k5,as.matrix(data.test))
c4<-compute(neuralnet4.k5,as.matrix(data.test))
c5<-compute(neuralnet5.k5,as.matrix(data.test))
c6<-compute(neuralnet6.k5,as.matrix(data.test))
neuralnet.result.k5 <- rep(0,dim(data.test)[1])
for(myVar in 1:dim(data.test)[1]){
  index <- which.max(c(c1$net.result[myVar],c2$net.result[myVar],c3$net.result[myVar],c4$net.result[myVar],c5$net.result[myVar],c6$net.result[myVar]))
  neuralnet.result.k5[myVar] <- index
}
neuralnet.perf.k5 <- table(neuralnet.result.k5,y.testfold)
nn.test.lda.error.k5 <- (1-sum(diag(neuralnet.perf.k5))/length(y.testfold))*100
nn.app.lda.error.k5 <- min(nn.lda.error.k5)

# ===
# Quadratic Discriminant Analysis
# ===
print("QDA")
# ACP
ind <- which.min(qda.acp.error)
qda.test.acp <- qda(y.app~., data=as.data.frame(X.acp.data[,1:ind]))
qda.test.acp.pred <- predict(qda.test.acp,newdata=as.data.frame(X.acp.test[,1:ind]))
qda.test.acp.perf <- table(y.test,qda.test.acp.pred$class)
qda.test.acp.error <- (1 - sum(diag(qda.test.acp.perf))/X.test.dim[1])*100
qda.app.acp.error <- min(qda.acp.error)

ind <- which.min(qda.acp.error.k5)
qda.test.acp.k5 <- qda(y.app~., data=as.data.frame(X.acp.data[,1:ind]))
qda.test.acp.pred.k5 <- predict(qda.test.acp.k5,newdata=as.data.frame(X.acp.test[,1:ind]))
qda.test.acp.perf.k5 <- table(y.test,qda.test.acp.pred.k5$class)
qda.test.acp.error.k5 <- (1 - sum(diag(qda.test.acp.perf.k5))/X.test.dim[1])*100
qda.app.acp.error.k5 <- min(qda.acp.error.k5)
  
# Forward
ind <- which.min(qda.forward.error)
qda.test.forward <- qda(y.app~., data=as.data.frame(X.forward.data[,1:ind]))
qda.test.forward.pred <- predict(qda.test.forward,newdata=as.data.frame(X.forward.test[,1:ind]))
qda.test.forward.perf <- table(y.test,qda.test.forward.pred$class)
qda.test.forward.error <- (1 - sum(diag(qda.test.forward.perf))/X.test.dim[1])*100
qda.app.forward.error <- min(qda.forward.error)

ind <- which.min(qda.forward.error.k5)
qda.test.forward.k5 <- qda(y.app~., data=as.data.frame(X.forward.data[,1:ind]))
qda.test.forward.pred.k5 <- predict(qda.test.forward.k5,newdata=as.data.frame(X.forward.test[,1:ind]))
qda.test.forward.perf.k5 <- table(y.test,qda.test.forward.pred.k5$class)
qda.test.forward.error.k5 <- (1 - sum(diag(qda.test.forward.perf.k5))/X.test.dim[1])*100
qda.app.forward.error.k5 <- min(qda.forward.error.k5)

# FDA
qda.test.lda <- qda(y.app~., data=as.data.frame(X.lda.data))
qda.test.lda.pred <- predict(qda.test.lda,newdata=as.data.frame(X.lda.test))
qda.test.lda.perf <- table(y.test,qda.test.lda.pred$class)
qda.test.lda.error <- (1 - sum(diag(qda.test.lda.perf))/X.test.dim[1])*100
qda.app.lda.error <- min(qda.lda.error)

qda.test.lda.k5 <- qda(y.app~., data=as.data.frame(X.lda.data))
qda.test.lda.pred.k5 <- predict(qda.test.lda.k5,newdata=as.data.frame(X.lda.test))
qda.test.lda.perf.k5 <- table(y.test,qda.test.lda.pred.k5$class)
qda.test.lda.error.k5 <- (1 - sum(diag(qda.test.lda.perf.k5))/X.test.dim[1])*100
qda.app.lda.error.k5 <- min(qda.lda.error.k5)


# Observation de nos meilleurs modeles maintenant qu'ils ont ete calcule

print(" =====    ACP     ======")
print(" =======================")
print("  5       |      10     ")
print(" =======================")
print(" QDA :")
cat("app :", qda.app.acp.error.k5, "\t", qda.app.acp.error, "\n")
cat("test :", qda.test.acp.error.k5,"\t", qda.test.acp.error,"\n")

print(" LDA :")
cat("app :", lda.app.acp.error.k5, "\t", lda.app.acp.error, "\n")
cat("test :", lda.test.acp.error.k5,"\t", lda.test.acp.error,"\n")

print(" KNN :")
cat("app :", knn.app.acp.error.k5, "\t", knn.app.acp.error, "\n")
cat("test :", knn.test.acp.error.k5,"\t", knn.test.acp.error,"\n")

print(" Log Reg :")
cat("app :", logReg.app.acp.error.k5, "\t", logReg.app.acp.error, "\n")
cat("test :", logReg.test.acp.error.k5,"\t", logReg.test.acp.error,"\n")

print(" NB :")
cat("app :", nb.app.acp.error.k5, "\t", nb.app.acp.error, "\n")
cat("test :", nb.test.acp.error.k5,"\t", nb.test.acp.error,"\n")

print(" SVM :")
cat("app :", svm.app.acp.error.k5, "\t", svm.app.acp.error, "\n")
cat("test :", svm.test.acp.error.k5,"\t", svm.test.acp.error,"\n")

print(" SVM Tune:")
cat("app :", svm.tune.app.acp.error.k5, "\t", svm.tune.app.acp.error, "\n")
cat("test :", svm.tune.test.acp.error.k5,"\t", svm.tune.test.acp.error,"\n")

print(" Tree :")
cat("app :", tree.app.acp.error.k5, "\t", tree.app.acp.error, "\n")
cat("test :", tree.test.acp.error.k5,"\t", tree.test.acp.error,"\n")

print(" RF :")
cat("app :", rf.app.acp.error.k5, "\t", rf.app.acp.error, "\n")
cat("test :", rf.test.acp.error.k5,"\t", rf.test.acp.error,"\n")

print(" NN :")
cat("app :", nn.app.acp.error.k5, "\t", nn.app.acp.error, "\n")
cat("test :", nn.test.acp.error.k5,"\t", nn.test.acp.error,"\n")



print(" ====    FORWARD     =====")
print(" =======================")
print("  5       |      10     ")
print(" =======================")
print(" QDA :")
cat("app :", qda.app.forward.error.k5, "\t", qda.app.forward.error, "\n")
cat("test :", qda.test.forward.error.k5,"\t", qda.test.forward.error,"\n")

print(" LDA :")
cat("app :", lda.app.forward.error.k5, "\t", lda.app.forward.error, "\n")
cat("test :", lda.test.forward.error.k5,"\t", lda.test.forward.error,"\n")

print(" KNN :")
cat("app :", knn.app.forward.error.k5, "\t", knn.app.forward.error, "\n")
cat("test :", knn.test.forward.error.k5,"\t", knn.test.forward.error,"\n")

print(" Log Reg :")
cat("app :", logReg.app.forward.error.k5, "\t", logReg.app.forward.error, "\n")
cat("test :", logReg.test.forward.error.k5,"\t", logReg.test.forward.error,"\n")

print(" NB :")
cat("app :", nb.app.forward.error.k5, "\t", nb.app.forward.error, "\n")
cat("test :", nb.test.forward.error.k5,"\t", nb.test.forward.error,"\n")

print(" SVM :")
cat("app :", svm.app.forward.error.k5, "\t", svm.app.forward.error, "\n")
cat("test :", svm.test.forward.error.k5,"\t", svm.test.forward.error,"\n")

print(" SVM Tune:")
cat("app :", svm.tune.app.forward.error.k5, "\t", svm.tune.app.forward.error, "\n")
cat("test :", svm.tune.test.forward.error.k5,"\t", svm.tune.test.forward.error,"\n")

print(" Tree :")
cat("app :", tree.app.forward.error.k5, "\t", tree.app.forward.error, "\n")
cat("test :", tree.test.forward.error.k5,"\t", tree.test.forward.error,"\n")

print(" RF :")
cat("app :", rf.app.forward.error.k5, "\t", rf.app.forward.error, "\n")
cat("test :", rf.test.forward.error.k5,"\t", rf.test.forward.error,"\n")

print(" NN :")
cat("app :", nn.app.forward.error.k5, "\t", nn.app.forward.error, "\n")
cat("test :", nn.test.forward.error.k5,"\t", nn.test.forward.error,"\n")




print(" =====    FDA     ======")
print(" =======================")
print("  5       |      10     ")
print(" =======================")
print(" QDA :")
cat("app :", qda.app.lda.error.k5, "\t", qda.app.lda.error, "\n")
cat("test :", qda.test.lda.error.k5,"\t", qda.test.lda.error,"\n")

print(" LDA :")
cat("app :", lda.app.lda.error.k5, "\t", lda.app.lda.error, "\n")
cat("test :", lda.test.lda.error.k5,"\t", lda.test.lda.error,"\n")

print(" KNN :")
cat("app :", knn.app.lda.error.k5, "\t", knn.app.lda.error, "\n")
cat("test :", knn.test.lda.error.k5,"\t", knn.test.lda.error,"\n")

print(" Log Reg :")
cat("app :", logReg.app.lda.error.k5, "\t", logReg.app.lda.error, "\n")
cat("test :", logReg.test.lda.error.k5,"\t", logReg.test.lda.error,"\n")

print(" NB :")
cat("app :", nb.app.lda.error.k5, "\t", nb.app.lda.error, "\n")
cat("test :", nb.test.lda.error.k5,"\t", nb.test.lda.error,"\n")

print(" SVM :")
cat("app :", svm.app.lda.error.k5, "\t", svm.app.lda.error, "\n")
cat("test :", svm.test.lda.error.k5,"\t", svm.test.lda.error,"\n")

print(" SVM Tune:")
cat("app :", svm.tune.app.lda.error.k5, "\t", svm.tune.app.lda.error, "\n")
cat("test :", svm.tune.test.lda.error.k5,"\t", svm.tune.test.lda.error,"\n")

print(" Tree :")
cat("app :", tree.app.lda.error.k5, "\t", tree.app.lda.error, "\n")
cat("test :", tree.test.lda.error.k5,"\t", tree.test.lda.error,"\n")

print(" RF :")
cat("app :", rf.app.lda.error.k5, "\t", rf.app.lda.error, "\n")
cat("test :", rf.test.lda.error.k5,"\t", rf.test.lda.error,"\n")

print(" NN :")
cat("app :", nn.app.lda.error.k5, "\t", nn.app.lda.error, "\n")
cat("test :", nn.test.lda.error.k5,"\t", nn.test.lda.error,"\n")