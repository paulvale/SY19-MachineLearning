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
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/end_final.rData")

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


# Enregistrement des erreurs de test
tree.test.acp.error <- rep(0,159)
tree.test.forward.error <- rep(0, 159)
tree.test.f.error <- rep(0, 159)

svm.test.acp.error <- rep(0,159)
svm.test.forward.error <- rep(0, 159)
svm.test.f.error <- rep(0, 159)

svm.tune.test.acp.error <- rep(0,159)
svm.tune.test.forward.error <- rep(0, 159)
svm.tune.test.f.error <- rep(0, 159)

nb.test.acp.error <- rep(0,159)
nb.test.forward.error <- rep(0,159)
nb.test.f.error <- rep(0,159)

logReg.test.acp.error <- rep(0,159)
logReg.test.forward.error <- rep(0, 159)
logReg.test.f.error <- rep(0, 159)

lda.test.acp.error <- rep(0,159)
lda.test.forward.error <- rep(0, 159)
lda.test.f.error <- rep(0, 159)

knn.test.acp.error <- rep(0,159)
knn.test.forward.error <- rep(0,159)
knn.test.f.error <- rep(0,159)

rf.test.acp.error <- rep(0,159)
rf.test.forward.error <- rep(0,159)
rf.test.f.error <- rep(0,159)

for(i in 2:160) {
  print(i)
  # ===
  # ACP
  # ===
  # Random Forest
  rf.test.acp <- randomForest(y.test.app.factor~., data=as.data.frame(X.acp.data[,1:i]), xtest=as.data.frame(X.acp.test[,1:i]), ytest=y.test.test.factor, ntree=600)
  rf.test.acp.error[i-1] <- (1 - (sum(diag(rf.test.acp$test$confusion)/X.test.dim[1])))*100
  
  # Tree
  tree.test.acp <- tree(factor(y.app)~., data=as.data.frame(X.acp.data[,1:i]))
  tree.test.acp.pred <- predict(tree.test.acp,newdata=as.data.frame(X.acp.test[,1:i]), type="class")
  tree.test.acp.perf <- table(y.test,factor(tree.test.acp.pred, levels=1:6))
  tree.test.acp.error[i-1] <- (1 - sum(diag(tree.test.acp.perf))/X.test.dim[1])*100
  
  # SVM
  svm.test.acp <- svm(X.acp.data[,1:i],y.app,type="C-classification")
  svm.test.acp.pred <- predict(svm.test.acp, X.acp.test[,1:i])
  svm.test.acp.perf <- table(factor(y.test),factor(svm.test.acp.pred, levels=1:6))
  svm.test.acp.error[i-1] <- (1 - sum(diag(svm.test.acp.perf))/X.test.dim[1])*100
  
  # SVM Tune
  svm.tune.test.acp <- tune(svm, train.y = factor(y.app),  train.x = X.acp.data[,1:i], cost=svm.tune.acp$best.parameters$cost, gamma=svm.tune.acp$best.parameters$gamma)
  svm.tune.test.acp.pred <- predict(svm.tune.test.acp$best.model , X.acp.test[,1:i])
  svm.tune.test.acp.perf <- table(factor(y.test),factor(svm.tune.test.acp.pred, levels=1:6))
  svm.tune.test.acp.error[i-1] <- (1 - sum(diag(svm.tune.test.acp.perf))/X.test.dim[1])*100
  
  # Naive Bayesien
  nb.test.acp <- naiveBayes(factor(y.app)~., data=as.data.frame(X.acp.data[,1:i]))
  nb.test.acp.pred <- predict(nb.test.acp,newdata=as.data.frame(X.acp.test[,1:i]))
  nb.test.acp.perf <- table(factor(y.test),factor(nb.test.acp.pred, levels=1:6))
  nb.test.acp.error[i-1] <- (1 - sum(diag(nb.test.acp.perf))/X.test.dim[1])*100
  
  # RegLog
  logReg.test.acp.res<-c(rep(0,X.test.dim[1]))
  logReg.test.acp <- glmnet(X.acp.data[,1:i] ,y=y.app,family="multinomial")
  logReg.test.acp.pred <- predict(logReg.test.acp,newx=X.acp.test[,1:i],type="response",s=logReg.test.acp$lambda.min)
  
  for (h in 1:X.test.dim[1]) {
    logReg.test.acp.res[h] <-which.max(logReg.test.acp.pred[h,1:6,dim(logReg.test.acp.pred)[3]-1])
  }
  logReg.test.acp.perf <- table(y.test.test.factor,factor(logReg.test.acp.res, levels=1:6))
  logReg.test.acp.error[i-1] <-(1 - sum(diag(logReg.test.acp.perf))/X.test.dim[1])*100
  
  # LDA
  lda.test.acp <- lda(y.app~., data=as.data.frame(X.acp.data[,1:i]))
  lda.test.acp.pred <- predict(lda.test.acp,newdata=as.data.frame(X.acp.test[,1:i]))
  lda.test.acp.perf <- table(y.test,lda.test.acp.pred$class)
  lda.test.acp.error[i-1] <- (1 - sum(diag(lda.test.acp.perf))/X.test.dim[1])*100
  
  # KNN
  knn.test.acp <- knn(as.data.frame(X.acp.data[,1:i]), as.data.frame(X.acp.test[,1:i]), y.app,k=25)
  knn.test.acp.perf <- table(y.test, factor(knn.test.acp, levels=1:6))
  knn.test.acp.error[i-1] <- (1 - sum(diag(knn.test.acp.perf))/X.test.dim[1])*100
  
  # ===
  # Forward on ACP
  # ===
  rf.test.forward <- randomForest(y.test.app.factor~., data=as.data.frame(X.forward.data[,1:i]), xtest=as.data.frame(X.forward.test[,1:i]), ytest=y.test.test.factor, ntree=100)
  rf.test.forward.error[i-1] <- (1 - (sum(diag(rf.test.forward$test$confusion)/X.test.dim[1])))*100
  
  # Tree
  tree.test.forward <- tree(factor(y.app)~., data=as.data.frame(X.forward.data[,1:i]))
  tree.test.forward.pred <- predict(tree.test.forward,newdata=as.data.frame(X.forward.test[,1:i]), type="class")
  tree.test.forward.perf <- table(y.test,factor(tree.test.forward.pred, levels=1:6))
  tree.test.forward.error[i-1] <- (1 - sum(diag(tree.test.forward.perf))/X.test.dim[1])*100
  
  # SVM
  svm.test.forward <- svm(X.forward.data[,1:i],y.app,type="C-classification")
  svm.test.forward.pred <- predict(svm.test.forward, X.forward.test[,1:i])
  svm.test.forward.perf <- table(factor(y.test),factor(svm.test.forward.pred, levels=1:6))
  svm.test.forward.error[i-1] <- (1 - sum(diag(svm.test.forward.perf))/X.test.dim[1])*100
  
  # SVM Tune
  svm.tune.test.forward <- tune(svm, train.y = factor(y.app),  train.x = X.forward.data[,1:i], cost=svm.tune.forward$best.parameters$cost, gamma=svm.tune.forward$best.parameters$gamma)
  svm.tune.test.forward.pred <- predict(svm.tune.test.forward$best.model , X.forward.test[,1:i])
  svm.tune.test.forward.perf <- table(factor(y.test),factor(svm.tune.test.forward.pred, levels=1:6))
  svm.tune.test.forward.error[i-1] <- (1 - sum(diag(svm.tune.test.forward.perf))/X.test.dim[1])*100
  
  # Naive Bayesien
  nb.test.forward <- naiveBayes(factor(y.app)~., data=as.data.frame(X.forward.data[,1:i]))
  nb.test.forward.pred <- predict(nb.test.forward,newdata=as.data.frame(X.forward.test[,1:i]))
  nb.test.forward.perf <- table(factor(y.test),factor(nb.test.forward.pred, levels=1:6))
  nb.test.forward.error[i-1] <- (1 - sum(diag(nb.test.forward.perf))/X.test.dim[1])*100
  
  # RegLog
  logReg.test.forward.res<-c(rep(0,X.test.dim[1]))
  logReg.test.forward <- glmnet(as.matrix(X.forward.data[,1:i]) ,y=y.app,family="multinomial")
  logReg.test.forward.pred <- predict(logReg.test.forward,newx=as.matrix(X.forward.test[,1:i]),type="response",s=logReg.test.forward$lambda.min)
  
  for (h in 1:X.test.dim[1]) {
    logReg.test.forward.res[h] <-which.max(logReg.test.forward.pred[h,1:6,dim(logReg.test.forward.pred)[3]-1])
  }
  logReg.test.forward.perf <- table(y.test,factor(logReg.test.forward.res, levels=1:6))
  logReg.test.forward.error[i-1] <-(1 - sum(diag(logReg.test.forward.perf))/X.test.dim[1])*100
  
  # LDA
  lda.test.forward <- lda(y.app~., data=as.data.frame(X.forward.data[,1:i]))
  lda.test.forward.pred <- predict(lda.test.forward,newdata=as.data.frame(X.forward.test[,1:i]))
  lda.test.forward.perf <- table(y.test,lda.test.forward.pred$class)
  lda.test.forward.error[i-1] <- (1 - sum(diag(lda.test.forward.perf))/X.test.dim[1])*100
  
  # KNN
  knn.test.forward <- knn(as.data.frame(X.forward.data[,1:i]), as.data.frame(X.forward.test[,1:i]), y.app,k=25)
  knn.test.forward.perf <- table(y.test,factor( knn.test.forward, levels=1:6))
  knn.test.forward.error[i-1] <- (1 - sum(diag(knn.test.forward.perf))/X.test.dim[1])*100
  
  # ===
  # Forward on Data
  # ===
  rf.test.f <- randomForest(y.test.app.factor~., data=as.data.frame(X.f.data[,1:i]), xtest=as.data.frame(X.f.test[,1:i]), ytest=y.test.test.factor, ntree=900)
  rf.test.f.error[i-1] <- (1 - (sum(diag(rf.test.f$test$confusion)/X.test.dim[1])))*100
  
  # Tree
  tree.test.f <- tree(factor(y.app)~., data=as.data.frame(X.f.data[,1:i]))
  tree.test.f.pred <- predict(tree.test.f,newdata=as.data.frame(X.f.test[,1:i]), type="class")
  tree.test.f.perf <- table(y.test,factor(tree.test.f.pred, levels=1:6))
  tree.test.f.error[i-1] <- (1 - sum(diag(tree.test.f.perf))/X.test.dim[1])*100
  
  # SVM
  svm.test.f <- svm(X.f.data[,1:i],y.app,type="C-classification")
  svm.test.f.pred <- predict(svm.test.f, X.f.test[,1:i])
  svm.test.f.perf <- table(factor(y.test),factor(svm.test.f.pred, levels=1:6))
  svm.test.f.error[i-1] <- (1 - sum(diag(svm.test.f.perf))/X.test.dim[1])*100
  
  # SVM Tune
  svm.tune.test.f <- tune(svm, train.y = factor(y.app),  train.x = X.f.data[,1:i], cost=svm.tune.f$best.parameters$cost, gamma=svm.tune.f$best.parameters$gamma)
  svm.tune.test.f.pred <- predict(svm.tune.test.f$best.model , X.f.test[,1:i])
  svm.tune.test.f.perf <- table(factor(y.test),factor(svm.tune.test.f.pred, levels=1:6))
  svm.tune.test.f.error[i-1] <- (1 - sum(diag(svm.tune.test.f.perf))/X.test.dim[1])*100
  
  # Naive Bayesien
  nb.test.f <- naiveBayes(factor(y.app)~., data=as.data.frame(X.f.data[,1:i]))
  nb.test.f.pred <- predict(nb.test.f,newdata=as.data.frame(X.f.test[,1:i]))
  nb.test.f.perf <- table(factor(y.test),factor(nb.test.f.pred, levels=1:6))
  nb.test.f.error[i-1] <- (1 - sum(diag(nb.test.f.perf))/X.test.dim[1])*100
  
  # RegLog
  logReg.test.f.res<-c(rep(0,X.test.dim[1]))
  logReg.test.f <- glmnet(as.matrix(X.f.data[,1:i]) ,y=y.app,family="multinomial")
  logReg.test.f.pred <- predict(logReg.test.f,newx=as.matrix(X.f.test[,1:i]),type="response",s=logReg.test.f$lambda.min)
  
  for (h in 1:X.test.dim[1]) {
    logReg.test.f.res[h] <-which.max(logReg.test.f.pred[h,1:6,dim(logReg.test.f.pred)[3]-1])
  }
  logReg.test.f.perf <- table(y.test,factor(logReg.test.f.res, levels=1:6))
  logReg.test.f.error[i-1] <-(1 - sum(diag(logReg.test.f.perf))/X.test.dim[1])*100
  
  # LDA
  lda.test.f <- lda(y.app~., data=as.data.frame(X.f.data[,1:i]))
  lda.test.f.pred <- predict(lda.test.f,newdata=as.data.frame(X.f.test[,1:i]))
  lda.test.f.perf <- table(y.test,lda.test.f.pred$class)
  lda.test.f.error[i-1] <- (1 - sum(diag(lda.test.f.perf))/X.test.dim[1])*100
  
  # KNN
  knn.test.f <- knn(as.data.frame(X.f.data[,1:i]), as.data.frame(X.f.test[,1:i]), y.app,k=20)
  knn.test.f.perf <- table(y.test,factor( knn.test.f, levels=1:6))
  knn.test.f.error[i-1] <- (1 - sum(diag(knn.test.f.perf))/X.test.dim[1])*100
}

# Observation des donnees
# KNN
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(knn.acp.error[25,], knn.test.acp.error))
max <- max(c(knn.acp.error[25,], knn.test.acp.error))
plot(knn.test.acp.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(knn.acp.error[25,], type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data / KNN")

min <- min(c(knn.forward.error[25,], knn.test.forward.error))
max <- max(c(knn.forward.error[25,], knn.test.forward.error))
plot(knn.test.forward.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(knn.forward.error[25,], type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward ACP / KNN")

min <- min(c(knn.f.error[20,], knn.test.f.error))
max <- max(c(knn.f.error[20,], knn.test.f.error))
plot(knn.test.f.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(knn.f.error[20,], type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data / KNN")


# RF
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(rf.acp.error[6,], rf.test.acp.error))
max <- max(c(rf.acp.error[6,], rf.test.acp.error))
plot(rf.test.acp.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(rf.acp.error[6,], type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data / RF")

min <- min(c(rf.forward.error[1,], rf.test.forward.error))
max <- max(c(rf.forward.error[1,], rf.test.forward.error))
plot(rf.test.forward.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(rf.forward.error[1,], type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward ACP / RF")

min <- min(c(rf.f.error[9,], rf.test.f.error))
max <- max(c(rf.f.error[9,], rf.test.f.error))
plot(rf.test.f.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(rf.f.error[9,], type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data / RF")

# LDA
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(lda.acp.error, lda.test.acp.error))
max <- max(c(lda.acp.error, lda.test.acp.error))
plot(lda.test.acp.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(lda.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data / LDA")

min <- min(c(lda.forward.error, lda.test.forward.error))
max <- max(c(lda.forward.error, lda.test.forward.error))
plot(lda.test.forward.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(lda.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward ACP / LDA")

min <- min(c(lda.f.error, lda.test.f.error))
max <- max(c(lda.f.error, lda.test.f.error))
plot(lda.test.f.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(lda.f.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data / LDA")

# LogReg
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(logReg.acp.error, logReg.test.acp.error))
max <- max(c(logReg.acp.error, logReg.test.acp.error))
plot(logReg.test.acp.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(logReg.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data / LogReg")

min <- min(c(logReg.forward.error, logReg.test.forward.error))
max <- max(c(logReg.forward.error, logReg.test.forward.error))
plot(logReg.test.forward.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(logReg.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward ACP / LogReg")

min <- min(c(logReg.f.error, logReg.test.f.error))
max <- max(c(logReg.f.error, logReg.test.f.error))
plot(logReg.test.f.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(logReg.f.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data / LogReg")

# NB
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(nb.acp.error, nb.test.acp.error))
max <- max(c(nb.acp.error, nb.test.acp.error))
plot(nb.test.acp.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(nb.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data / NB")

min <- min(c(nb.forward.error, nb.test.forward.error))
max <- max(c(nb.forward.error, nb.test.forward.error))
plot(nb.test.forward.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(nb.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward ACP / NB")

min <- min(c(nb.f.error, nb.test.f.error))
max <- max(c(nb.f.error, nb.test.f.error))
plot(nb.test.f.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(nb.f.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data / NB")

# SVM
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(svm.acp.error, svm.test.acp.error))
max <- max(c(svm.acp.error, svm.test.acp.error))
plot(svm.test.acp.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(svm.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data / SVM")

min <- min(c(svm.forward.error, svm.test.forward.error))
max <- max(c(svm.forward.error, svm.test.forward.error))
plot(svm.test.forward.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(svm.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward ACP / SVM")

min <- min(c(svm.f.error, svm.test.f.error))
max <- max(c(svm.f.error, svm.test.f.error))
plot(svm.test.f.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(svm.f.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data / SVM")

# SVM Tune
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(svm.tune.acp.error, svm.tune.test.acp.error))
max <- max(c(svm.tune.acp.error, svm.tune.test.acp.error))
plot(svm.tune.test.acp.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(svm.tune.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data / SVM Tune")

min <- min(c(svm.tune.forward.error, svm.tune.test.forward.error))
max <- max(c(svm.tune.forward.error, svm.tune.test.forward.error))
plot(svm.tune.test.forward.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(svm.tune.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward ACP / SVM Tune")

min <- min(c(svm.tune.f.error, svm.tune.test.f.error))
max <- max(c(svm.tune.f.error, svm.tune.test.f.error))
plot(svm.tune.test.f.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(svm.tune.f.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data / SVM Tune")

# Tree
# Evolution en fonction du nombre de composantes prises en comptes
min <- min(c(tree.acp.error, tree.test.acp.error))
max <- max(c(tree.acp.error, tree.test.acp.error))
plot(tree.test.acp.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(tree.acp.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with ACP Data / Tree")

min <- min(c(tree.forward.error, tree.test.forward.error))
max <- max(c(tree.test.forward.error, tree.forward.error))
plot(tree.test.forward.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(tree.forward.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward ACP / Tree")

min <- min(c(tree.f.error, tree.test.f.error))
max <- max(c(tree.test.f.error, tree.f.error))
plot(tree.test.f.error,type="l", col="red", ylim=c(min,max), ylab="", xlab="")
par(new=TRUE)
plot(tree.f.error, type="l", xlab="number of principal components", ylab="MSE",ylim=c(min,max), main="Evolution of error with Forward Data / Tree")

