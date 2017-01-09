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
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/end_10_forward.rData")

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

# 3) Forward/Backward Selection on ACP
reg.fit<-regsubsets(y.app~.,data=as.data.frame(X.acp.data),method="forward")
vorder <- reg.fit$vorder[2:length(reg.fit$vorder)]
vorder <- vorder - 1
X.forward.data <- X.acp.data[,vorder[1]]
X.forward.data <- as.data.frame(X.forward.data)
names(X.forward.data)[1] <- colnames(X.acp.data)[vorder[1]]

# 4) Forward/Backward Selection on Data
reg.fit2 <- regsubsets(y.app~., data=as.data.frame(X.app), method="forward")
vorder2 <- reg.fit2$vorder[2:length(reg.fit2$vorder)]
vorder2 <- vorder2 - 1
X.f.data <- X.app[,vorder2[1]]
X.f.data <- as.data.frame(X.f.data)
names(X.f.data)[1] <- "V1"
for(i in 2:160){
  X.f.data[i] <- X.app[,vorder2[i]]
}

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

# 3) Forward on ACP
X.forward.test <- X.acp.test[,vorder[1]]
X.forward.test <- as.data.frame(X.forward.test)
names(X.forward.test)[1] <- colnames(X.acp.test)[vorder[1]]

# 4) Forward on Data
X.f.test <- X.test[,vorder2[1]]
X.f.test <- as.data.frame(X.f.test)
names(X.f.test)[1] <- "V1"

for(j in 2:160){
  X.forward.data[[colnames(X.acp.data)[vorder[j]]]] <- X.acp.data[,vorder[j]]
  X.forward.test[[colnames(X.acp.test)[vorder[j]]]] <- X.acp.test[,vorder[j]]
  
  X.f.data[j] <- X.app[,vorder2[j]]
  X.f.test[j] <- X.test[,vorder2[j]]
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

# Tree
ind <- which.min(tree.acp.error)
tree.test.acp <- tree(factor(y.app)~., data=as.data.frame(X.acp.data[,1:ind]))
tree.test.acp.pred <- predict(tree.test.acp,newdata=as.data.frame(X.acp.test[,1:ind]), type="class")
tree.test.acp.perf <- table(y.test,factor(tree.test.acp.pred, levels=1:6))
tree.test.acp.error <- (1 - sum(diag(tree.test.acp.perf))/X.test.dim[1])*100
tree.app.acp.error <- min(tree.acp.error)

# SVM
ind <- which.min(svm.acp.error)
svm.test.acp <- svm(X.acp.data[,1:ind],y.app,type="C-classification")
svm.test.acp.pred <- predict(svm.test.acp, X.acp.test[,1:ind])
svm.test.acp.perf <- table(factor(y.test),factor(svm.test.acp.pred, levels=1:6))
svm.test.acp.error <- (1 - sum(diag(svm.test.acp.perf))/X.test.dim[1])*100
svm.app.acp.error <- min(svm.acp.error)

# SVM Tune
ind <- which.min(svm.tune.acp.error)
svm.tune.test.acp <- tune(svm, train.y = factor(y.app),  train.x = X.acp.data[,1:ind], cost=svm.tune.acp$best.parameters$cost, gamma=svm.tune.acp$best.parameters$gamma)
svm.tune.test.acp.pred <- predict(svm.tune.test.acp$best.model , X.acp.test[,1:ind])
svm.tune.test.acp.perf <- table(factor(y.test),factor(svm.tune.test.acp.pred, levels=1:6))
svm.tune.test.acp.error <- (1 - sum(diag(svm.tune.test.acp.perf))/X.test.dim[1])*100
svm.tune.app.acp.error <- min(svm.tune.acp.error)

# Naive Bayesien
ind <- which.min(nb.acp.error)
nb.test.acp <- naiveBayes(factor(y.app)~., data=as.data.frame(X.acp.data[,1:ind]))
nb.test.acp.pred <- predict(nb.test.acp,newdata=as.data.frame(X.acp.test[,1:ind]))
nb.test.acp.perf <- table(factor(y.test),factor(nb.test.acp.pred, levels=1:6))
nb.test.acp.error <- (1 - sum(diag(nb.test.acp.perf))/X.test.dim[1])*100
nb.app.acp.error <- min(nb.acp.error)

# RegLog
ind <- which.min(logReg.acp.error)
logReg.test.acp.res<-c(rep(0,X.test.dim[1]))
logReg.test.acp <- glmnet(X.acp.data[,1:ind] ,y=y.app,family="multinomial")
logReg.test.acp.pred <- predict(logReg.test.acp,newx=X.acp.test[,1:ind],type="response",s=logReg.test.acp$lambda.min)

for (h in 1:X.test.dim[1]) {
  logReg.test.acp.res[h] <-which.max(logReg.test.acp.pred[h,1:6,dim(logReg.test.acp.pred)[3]-1])
}
logReg.test.acp.perf <- table(y.test.test.factor,factor(logReg.test.acp.res, levels=1:6))
logReg.test.acp.error <-(1 - sum(diag(logReg.test.acp.perf))/X.test.dim[1])*100
logReg.app.acp.error <- min(logReg.acp.error)

# LDA
ind <- which.min(lda.acp.error)
lda.test.acp <- lda(y.app~., data=as.data.frame(X.acp.data[,1:ind]))
lda.test.acp.pred <- predict(lda.test.acp,newdata=as.data.frame(X.acp.test[,1:ind]))
lda.test.acp.perf <- table(y.test,lda.test.acp.pred$class)
lda.test.acp.error <- (1 - sum(diag(lda.test.acp.perf))/X.test.dim[1])*100
lda.app.acp.error <- min(lda.acp.error)

# KNN
ind <-   which(knn.acp.error == min(knn.acp.error), arr.ind = TRUE) 
knn.test.acp <- knn(as.data.frame(X.acp.data[,1:ind[2]]), as.data.frame(X.acp.test[,1:ind[2]]), y.app,k=ind)
knn.test.acp.perf <- table(y.test, factor(knn.test.acp, levels=1:6))
knn.test.acp.error <- (1 - sum(diag(knn.test.acp.perf))/X.test.dim[1])*100
knn.app.acp.error <- min(knn.acp.error)


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
neuralnet.perf <- table(factor(neuralnet.result, levels=1:6),y.testfold)
nn.test.acp.error <- (1-sum(diag(neuralnet.perf))/length(y.testfold))*100
nn.app.acp.error <- min(nn.acp.error)

# ===
# Forward on ACP
# ===
print("Forward on ACP")
ind <-  which(rf.forward.error == min(rf.forward.error), arr.ind = TRUE) 
numberofTree <- vectorTree[ind[1]]
rf.test.forward <- randomForest(y.test.app.factor~., data=as.data.frame(X.forward.data[,1:ind[2]]), xtest=as.data.frame(X.forward.test[,1:ind[2]]), ytest=y.test.test.factor, ntree=numberofTree)
rf.test.forward.error <- (1 - (sum(diag(rf.test.forward$test$confusion)/X.test.dim[1])))*100
rf.app.forward.error <- min(rf.forward.error)

# Tree
ind <- which.min(tree.forward.error)
tree.test.forward <- tree(factor(y.app)~., data=as.data.frame(X.forward.data[,1:ind]))
tree.test.forward.pred <- predict(tree.test.forward,newdata=as.data.frame(X.forward.test[,1:ind]), type="class")
tree.test.forward.perf <- table(y.test,factor(tree.test.forward.pred, levels=1:6))
tree.test.forward.error <- (1 - sum(diag(tree.test.forward.perf))/X.test.dim[1])*100
tree.app.forward.error <- min(tree.forward.error)

# SVM
ind <- which.min(svm.forward.error)
svm.test.forward <- svm(X.forward.data[,1:ind],y.app,type="C-classification")
svm.test.forward.pred <- predict(svm.test.forward, X.forward.test[,1:ind])
svm.test.forward.perf <- table(factor(y.test),factor(svm.test.forward.pred, levels=1:6))
svm.test.forward.error <- (1 - sum(diag(svm.test.forward.perf))/X.test.dim[1])*100
svm.app.forward.error <- min(svm.forward.error)

# SVM Tune
ind <- which.min(svm.tune.forward.error)
svm.tune.test.forward <- tune(svm, train.y = factor(y.app),  train.x = X.forward.data[,1:ind], cost=svm.tune.forward$best.parameters$cost, gamma=svm.tune.forward$best.parameters$gamma)
svm.tune.test.forward.pred <- predict(svm.tune.test.forward$best.model , X.forward.test[,1:ind])
svm.tune.test.forward.perf <- table(factor(y.test),factor(svm.tune.test.forward.pred, levels=1:6))
svm.tune.test.forward.error <- (1 - sum(diag(svm.tune.test.forward.perf))/X.test.dim[1])*100
svm.tune.app.forward.error <- min(svm.tune.forward.error)

# Naive Bayesien
ind <- which.min(nb.forward.error)
nb.test.forward <- naiveBayes(factor(y.app)~., data=as.data.frame(X.forward.data[,1:ind]))
nb.test.forward.pred <- predict(nb.test.forward,newdata=as.data.frame(X.forward.test[,1:ind]))
nb.test.forward.perf <- table(factor(y.test),factor(nb.test.forward.pred, levels=1:6))
nb.test.forward.error <- (1 - sum(diag(nb.test.forward.perf))/X.test.dim[1])*100
nb.app.forward.error <- min(nb.forward.error)

# RegLog
ind <- which.min(logReg.forward.error)
logReg.test.forward.res<-c(rep(0,X.test.dim[1]))
logReg.test.forward <- glmnet(as.matrix(X.forward.data[,1:ind]) ,y=y.app,family="multinomial")
logReg.test.forward.pred <- predict(logReg.test.forward,newx=as.matrix(X.forward.test[,1:ind]),type="response",s=logReg.test.forward$lambda.min)

for (h in 1:X.test.dim[1]) {
  logReg.test.forward.res[h] <-which.max(logReg.test.forward.pred[h,1:6,dim(logReg.test.forward.pred)[3]-1])
}
logReg.test.forward.perf <- table(y.test,factor(logReg.test.forward.res, levels=1:6))
logReg.test.forward.error <-(1 - sum(diag(logReg.test.forward.perf))/X.test.dim[1])*100
logReg.app.forward.error <- min(logReg.forward.error)

# LDA
ind <- which.min(lda.forward.error)
lda.test.forward <- lda(y.app~., data=as.data.frame(X.forward.data[,1:ind]))
lda.test.forward.pred <- predict(lda.test.forward,newdata=as.data.frame(X.forward.test[,1:ind]))
lda.test.forward.perf <- table(y.test,lda.test.forward.pred$class)
lda.test.forward.error <- (1 - sum(diag(lda.test.forward.perf))/X.test.dim[1])*100
lda.app.forward.error <- min(lda.forward.error)

# KNN
ind <-  which(knn.forward.error == min(knn.forward.error), arr.ind = TRUE) 
knn.test.forward <- knn(as.data.frame(X.forward.data[,1:ind[2]]), as.data.frame(X.forward.test[,1:ind[2]]), y.app,k=ind)
knn.test.forward.perf <- table(y.test,factor( knn.test.forward, levels=1:6))
knn.test.forward.error <- (1 - sum(diag(knn.test.forward.perf))/X.test.dim[1])*100
knn.app.forward.error <- min(knn.forward.error)

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
neuralnet.perf <- table(factor(neuralnet.result, levels=1:6),y.testfold)
nn.test.forward.error <- (1-sum(diag(neuralnet.perf))/length(y.testfold))*100
nn.app.forward.error <- min(nn.forward.error)


# ===
# Forward on Data
# ===
print("Forward on Data")
ind <-  which(rf.f.error == min(rf.f.error), arr.ind = TRUE) 
numberofTree <- vectorTree[ind[1]]
rf.test.f <- randomForest(y.test.app.factor~., data=as.data.frame(X.f.data[,1:ind[2]]), xtest=as.data.frame(X.f.test[,1:ind[2]]), ytest=y.test.test.factor, ntree=numberofTree)
rf.test.f.error <- (1 - (sum(diag(rf.test.f$test$confusion)/X.test.dim[1])))*100
rf.app.f.error <- min(rf.f.error)

# Tree
ind <- which.min(tree.f.error)
tree.test.f <- tree(factor(y.app)~., data=as.data.frame(X.f.data[,1:ind]))
tree.test.f.pred <- predict(tree.test.f,newdata=as.data.frame(X.f.test[,1:ind]), type="class")
tree.test.f.perf <- table(y.test,factor(tree.test.f.pred, levels=1:6))
tree.test.f.error <- (1 - sum(diag(tree.test.f.perf))/X.test.dim[1])*100
tree.app.f.error <- min(tree.f.error)

# SVM
ind <- which.min(svm.f.error)
svm.test.f <- svm(X.f.data[,1:ind],y.app,type="C-classification")
svm.test.f.pred <- predict(svm.test.f, X.f.test[,1:ind])
svm.test.f.perf <- table(factor(y.test),factor(svm.test.f.pred, levels=1:6))
svm.test.f.error <- (1 - sum(diag(svm.test.f.perf))/X.test.dim[1])*100
svm.app.f.error <- min(svm.f.error)

# SVM Tune
ind <- which.min(svm.tune.f.error)
svm.tune.test.f <- tune(svm, train.y = factor(y.app),  train.x = X.f.data[,1:ind], cost=svm.tune.f$best.parameters$cost, gamma=svm.tune.f$best.parameters$gamma)
svm.tune.test.f.pred <- predict(svm.tune.test.f$best.model , X.f.test[,1:ind])
svm.tune.test.f.perf <- table(factor(y.test),factor(svm.tune.test.f.pred, levels=1:6))
svm.tune.test.f.error <- (1 - sum(diag(svm.tune.test.f.perf))/X.test.dim[1])*100
svm.tune.app.f.error <- min(svm.tune.f.error)

# Naive Bayesien
ind <- which.min(nb.f.error)
nb.test.f <- naiveBayes(factor(y.app)~., data=as.data.frame(X.f.data[,1:ind]))
nb.test.f.pred <- predict(nb.test.f,newdata=as.data.frame(X.f.test[,1:ind]))
nb.test.f.perf <- table(factor(y.test),factor(nb.test.f.pred, levels=1:6))
nb.test.f.error <- (1 - sum(diag(nb.test.f.perf))/X.test.dim[1])*100
nb.app.f.error <- min(nb.f.error)

# RegLog
ind <- which.min(logReg.f.error)
logReg.test.f.res<-c(rep(0,X.test.dim[1]))
logReg.test.f <- glmnet(as.matrix(X.f.data[,1:ind]) ,y=y.app,family="multinomial")
logReg.test.f.pred <- predict(logReg.test.f,newx=as.matrix(X.f.test[,1:ind]),type="response",s=logReg.test.f$lambda.min)

for (h in 1:X.test.dim[1]) {
  logReg.test.f.res[h] <-which.max(logReg.test.f.pred[h,1:6,dim(logReg.test.f.pred)[3]-1])
}
logReg.test.f.perf <- table(y.test,factor(logReg.test.f.res, levels=1:6))
logReg.test.f.error <-(1 - sum(diag(logReg.test.f.perf))/X.test.dim[1])*100
logReg.app.f.error <- min(logReg.f.error)

# LDA
# ind <- which.min(lda.f.error)
lda.test.f <- lda(y.app~., data=as.data.frame(X.f.data[,1:15]))
lda.test.f.pred <- predict(lda.test.f,newdata=as.data.frame(X.f.test[,1:15]))
lda.test.f.perf <- table(y.test,lda.test.f.pred$class)
lda.test.f.error <- (1 - sum(diag(lda.test.f.perf))/X.test.dim[1])*100
lda.app.f.error <- lda.f.error[15]

# KNN
ind <-  which(knn.f.error == min(knn.f.error), arr.ind = TRUE) 
knn.test.f <- knn(as.data.frame(X.f.data[,1:ind[2]]), as.data.frame(X.f.test[,1:ind[2]]), y.app,k=ind)
knn.test.f.perf <- table(y.test,factor( knn.test.f, levels=1:6))
knn.test.f.error <- (1 - sum(diag(knn.test.f.perf))/X.test.dim[1])*100
knn.app.f.error <- min(knn.f.error)

# Neural Network
ind <- which.min(nn.f.error)
data.train = X.f.data[,1:ind]
data.test = X.f.test[,1:ind]
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
neuralnet.perf <- table(factor(neuralnet.result, levels=1:6),y.testfold)
nn.test.f.error <- (1-sum(diag(neuralnet.perf))/length(y.testfold))*100
nn.app.f.error <- min(nn.f.error)


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

# Tree
tree.test.lda <- tree(factor(y.app)~., data=as.data.frame(X.lda.data))
tree.test.lda.pred <- predict(tree.test.lda,newdata=as.data.frame(X.lda.test), type="class")
tree.test.lda.perf <- table(y.test,factor(tree.test.lda.pred, levels=1:6))
tree.test.lda.error <- (1 - sum(diag(tree.test.lda.perf))/X.test.dim[1])*100
tree.app.lda.error <- min(tree.lda.error)

# SVM
svm.test.lda <- svm(X.lda.data,y.app,type="C-classification")
svm.test.lda.pred <- predict(svm.test.lda, X.lda.test)
svm.test.lda.perf <- table(factor(y.test),factor(svm.test.lda.pred, levels=1:6))
svm.test.lda.error <- (1 - sum(diag(svm.test.lda.perf))/X.test.dim[1])*100
svm.app.lda.error <- min(svm.lda.error)

# SVM Tune
svm.tune.test.lda <- tune(svm, train.y = factor(y.app),  train.x = X.lda.data, cost=svm.tune.lda$best.parameters$cost, gamma=svm.tune.lda$best.parameters$gamma)
svm.tune.test.lda.pred <- predict(svm.tune.test.lda$best.model , X.lda.test)
svm.tune.test.lda.perf <- table(factor(y.test),factor(svm.tune.test.lda.pred, levels=1:6))
svm.tune.test.lda.error <- (1 - sum(diag(svm.tune.test.lda.perf))/X.test.dim[1])*100
svm.tune.app.lda.error <- min(svm.tune.lda.error)

# Naive Bayesien
nb.test.lda <- naiveBayes(factor(y.app)~., data=as.data.frame(X.lda.data))
nb.test.lda.pred <- predict(nb.test.lda,newdata=as.data.frame(X.lda.test))
nb.test.lda.perf <- table(factor(y.test),factor(nb.test.lda.pred, levels=1:6))
nb.test.lda.error <- (1 - sum(diag(nb.test.lda.perf))/X.test.dim[1])*100
nb.app.lda.error <- min(nb.lda.error)

# RegLog
logReg.test.lda.res<-c(rep(0,X.test.dim[1]))
logReg.test.lda <- glmnet(X.lda.data ,y=y.app,family="multinomial")
logReg.test.lda.pred <- predict(logReg.test.lda,newx=X.lda.test,type="response",s=logReg.test.lda$lambda.min)

for (h in 1:X.test.dim[1]) {
  logReg.test.lda.res[h] <-which.max(logReg.test.lda.pred[h,1:6,dim(logReg.test.lda.pred)[3]-1])
}
logReg.test.lda.perf <- table(y.test,factor(logReg.test.lda.res, levels=1:6))
logReg.test.lda.error <-(1 - sum(diag(logReg.test.lda.perf))/X.test.dim[1])*100
logReg.app.lda.error <- min(logReg.lda.error)

# LDA
lda.test.lda <- lda(y.app~., data=as.data.frame(X.lda.data))
lda.test.lda.pred <- predict(lda.test.lda,newdata=as.data.frame(X.lda.test))
lda.test.lda.perf <- table(y.test,lda.test.lda.pred$class)
lda.test.lda.error <- (1 - sum(diag(lda.test.lda.perf))/X.test.dim[1])*100
lda.app.lda.error <- min(lda.lda.error)

# KNN
ind <-  which.min(knn.lda.error)
knn.test.lda <- knn(as.data.frame(X.lda.data), as.data.frame(X.lda.test), y.app,k=ind)
knn.test.lda.perf <- table(y.test, factor(knn.test.lda, levels=1:6))
knn.test.lda.error <- (1 - sum(diag(knn.test.lda.perf))/X.test.dim[1])*100
knn.app.lda.error <- min(knn.lda.error)

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
neuralnet.perf <- table(factor(neuralnet.result, levels=1:6),y.testfold)
nn.test.lda.error <- (1-sum(diag(neuralnet.perf))/length(y.testfold))*100
nn.app.lda.error <- min(nn.lda.error)


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

# Forward on ACP
ind <- which.min(qda.forward.error)
qda.test.forward <- qda(y.app~., data=as.data.frame(X.forward.data[,1:ind]))
qda.test.forward.pred <- predict(qda.test.forward,newdata=as.data.frame(X.forward.test[,1:ind]))
qda.test.forward.perf <- table(y.test,qda.test.forward.pred$class)
qda.test.forward.error <- (1 - sum(diag(qda.test.forward.perf))/X.test.dim[1])*100
qda.app.forward.error <- min(qda.forward.error)

# Forward on Data
ind <- which.min(qda.f.error)
qda.test.f <- qda(y.app~., data=as.data.frame(X.f.data[,1:ind]))
qda.test.f.pred <- predict(qda.test.f,newdata=as.data.frame(X.f.test[,1:ind]))
qda.test.f.perf <- table(y.test,qda.test.f.pred$class)
qda.test.f.error <- (1 - sum(diag(qda.test.f.perf))/X.test.dim[1])*100
qda.app.f.error <- min(qda.f.error)

# FDA
qda.test.lda <- qda(y.app~., data=as.data.frame(X.lda.data))
qda.test.lda.pred <- predict(qda.test.lda,newdata=as.data.frame(X.lda.test))
qda.test.lda.perf <- table(y.test,qda.test.lda.pred$class)
qda.test.lda.error <- (1 - sum(diag(qda.test.lda.perf))/X.test.dim[1])*100
qda.app.lda.error <- min(qda.lda.error)


# Observation de nos meilleurs modeles maintenant qu'ils ont ete calcule


print(" =======================")
print("   ACP       |      Forward On ACP |  Forward  on Data | FDA      ")
print(" =======================")
print(" QDA :")
cat("app :",qda.app.acp.error, "\t",qda.app.forward.error, "\t",qda.app.f.error, "\t",qda.app.lda.error,"\n")
cat("test :",qda.test.acp.error,"\t",qda.test.forward.error,"\t",qda.test.f.error,"\t",qda.test.lda.error,"\n")

print(" LDA :")
cat("app :",  lda.app.acp.error, "\t",lda.app.forward.error, "\t",lda.app.f.error, "\t",lda.app.lda.error,"\n")
cat("test :",  lda.test.acp.error,"\t",lda.test.forward.error,"\t",lda.test.f.error,"\t",lda.test.lda.error,"\n")

print(" KNN :")
cat("app :", knn.app.acp.error, "\t",knn.app.forward.error, "\t",knn.app.f.error, "\t",knn.app.lda.error, "\n")
cat("test :",  knn.test.acp.error,"\t",knn.test.forward.error,"\t",knn.test.f.error,"\t",knn.test.lda.error,"\n")

print(" Log Reg :")
cat("app :",logReg.app.acp.error, "\t",logReg.app.forward.error, "\t",logReg.app.f.error, "\t", logReg.app.lda.error, "\n")
cat("test :", logReg.test.acp.error,"\t",logReg.test.forward.error,"\t",logReg.test.f.error,"\t",logReg.test.lda.error,"\n")

print(" NB :")
cat("app :", nb.app.acp.error, "\t",nb.app.forward.error, "\t",nb.app.f.error, "\t",nb.app.lda.error, "\n")
cat("test :", nb.test.acp.error,"\t", nb.test.acp.error,"\t", nb.test.f.error,"\t",nb.test.lda.error,"\n")

print(" SVM :")
cat("app :", svm.app.acp.error, "\t", svm.app.forward.error, "\t", svm.app.f.error, "\t",svm.app.lda.error, "\n")
cat("test :", svm.test.acp.error,"\t",svm.test.forward.error,"\t",svm.test.f.error,"\t",svm.test.lda.error,"\n")

print(" SVM Tune:")
cat("app :", svm.tune.app.acp.error, "\t",svm.tune.app.forward.error, "\t",svm.tune.app.f.error, "\t",svm.tune.app.lda.error, "\n")
cat("test :", svm.tune.test.acp.error,"\t",svm.tune.test.forward.error, "\t",svm.tune.test.f.error, "\t",svm.tune.test.lda.error, "\n")

print(" Tree :")
cat("app :",tree.app.acp.error, "\t",tree.app.forward.error, "\t",tree.app.f.error, "\t",tree.app.lda.error, "\n")
cat("test :", tree.test.acp.error,"\t",tree.test.forward.error,"\t",tree.test.f.error,"\t",tree.test.lda.error,"\n")

print(" RF :")
cat("app :", rf.app.acp.error, "\t",rf.app.forward.error, "\t",rf.app.f.error, "\t",rf.app.lda.error, "\n")
cat("test :", rf.test.acp.error,"\t",rf.test.forward.error,"\t",rf.test.f.error,"\t",rf.test.lda.error,"\n")

print(" NN :")
cat("app :", nn.app.acp.error, "\t",nn.app.forward.error, "\t",nn.app.f.error, "\t",nn.app.lda.error, "\n")
cat("test :", nn.test.acp.error,"\t", nn.test.forward.error,"\t", nn.test.f.error,"\t", nn.test.lda.error,"\n")