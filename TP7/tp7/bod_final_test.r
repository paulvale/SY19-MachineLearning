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
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/endNN.rData")

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
svm.tune.acp.error.k5 <- svm.tune.acp.error
svm.tune.lda.error.k5 <- svm.tune.lda.error
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
methodArray <- c("qda.acp.error.k5","qda.lda.error.k5","qda.forward.error.k5",
                 "lda.acp.error.k5","lda.lda.error.k5","lda.forward.error.k5",
                 "knn.acp.error.k5","knn.lda.error.k5","knn.forward.error.k5",
                 "logReg.acp.error.k5","logReg.lda.error.k5","logReg.forward.error.k5",
                 "nb.acp.error.k5","nb.lda.error.k5","nb.forward.error.k5",
                 "svm.acp.error.k5","svm.lda.error.k5","svm.forward.error.k5",
                 "svm.tune.acp.error.k5","svm.tune.lda.error.k5","svm.tune.forward.error.k5",
                 "tree.acp.error.k5","tree.lda.error.k5","tree.forward.error.k5",
                 "rf.acp.error.k5","rf.lda.error.k5","rf.forward.error.k5",
                 "nn.acp.error.k5","nn.lda.error.k5","nn.forward.error.k5",
                 "qda.acp.error","qda.lda.error","qda.forward.error",
                 "lda.acp.error","lda.lda.error","lda.forward.error",
                 "knn.acp.error","knn.lda.error","knn.forward.error",
                 "logReg.acp.error","logReg.lda.error","logReg.forward.error",
                 "nb.acp.error","nb.lda.error","nb.forward.error",
                 "svm.acp.error","svm.lda.error","svm.forward.error",
                 "svm.tune.acp.error","svm.tune.lda.error","svm.tune.forward.error",
                 "tree.acp.error","tree.lda.error","tree.forward.error",
                 "rf.acp.error","rf.lda.error","rf.forward.error",
                 "nn.acp.error","nn.lda.error","nn.forward.error"
                )

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

methodMatrix <- matrix(data = methodArray, ncol=3, byrow = TRUE)

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


for(i in 1:dim(methodMatrix)[1]){
  acp <- get(methodMatrix[i,1])
  lda <- get(methodMatrix[i,2])
  forward <- get(methodMatrix[i,3])
  
  acp.min <- min(acp)
  lda.min <- min(lda)
  forward.min <- min(forward)
  
  if(acp.min < lda.min && acp.min < forward.min){
    print("acp")
    ind =  which(acp == min(acp), arr.ind = TRUE)
    print(ind)
  } else if(lda.min < acp.min && lda.min < forward.min){
    if( i == 3 ){
      ind =  which(lda == min(lda), arr.ind = TRUE) 
      knn.test.lda.k5 <- knn(as.data.frame(X.lda.data), as.data.frame(X.lda.test), y.app,k=ind[1])
      knn.test.lda.perf.k5 <- table(y.test, knn.test.lda.k5)
      knn.test.lda.error.k5 <- (1 - sum(diag(knn.test.lda.perf.k5))/X.test.dim[1])*100
      knn.app.lda.error.k5 <- lda.min
    } else if( i == 13 ){
      ind =  which(lda == min(lda), arr.ind = TRUE) 
      knn.test.lda <- knn(as.data.frame(X.lda.data), as.data.frame(X.lda.test), y.app,k=ind[1])
      knn.test.lda.perf <- table(y.test, knn.test.lda)
      knn.test.lda.error <- (1 - sum(diag(knn.test.lda.perf))/X.test.dim[1])*100
      knn.app.lda.error <- lda.min
    } else if (i == 9){
      ind =  which(lda == min(lda), arr.ind = TRUE) 
      numberofTree <- vectorTree[ind]
      rf.test.lda.k5 <- randomForest(y.test.app.factor~., data=as.data.frame(X.lda.data), xtest=as.data.frame(X.lda.test), ytest=y.test.test.factor, ntree=numberofTree)
      rf.test.lda.error.k5 <- (1 - (sum(diag(rf.test.lda.k5$test$confusion)/X.test.dim[1])))*100
      rf.app.lda.error.k5 <- lda.min
    } else if (i == 19){
      ind =  which(lda == min(lda), arr.ind = TRUE) 
      numberofTree <- vectorTree[ind]
      rf.test.lda <- randomForest(y.test.app.factor~., data=as.data.frame(X.lda.data), xtest=as.data.frame(X.lda.test), ytest=y.test.test.factor, ntree=numberofTree)
      rf.test.lda.error <- (1 - (sum(diag(rf.test.lda$test$confusion)/X.test.dim[1])))*100
      rf.app.lda.error <- lda.min
    } else {
      if( i == 1) { # QDA
        qda.test.lda.k5 <- qda(y.app~., data=as.data.frame(X.lda.data))
        qda.test.lda.pred.k5 <- predict(qda.test.lda.k5,newdata=as.data.frame(X.lda.test))
        qda.test.lda.perf.k5 <- table(y.test,qda.test.lda.pred.k5$class)
        qda.test.lda.error.k5 <- (1 - sum(diag(qda.test.lda.perf.k5))/X.test.dim[1])*100
        qda.app.lda.error.k5 <- lda.min
      }else if( i == 11) { # QDA
        qda.test.lda <- qda(y.app~., data=as.data.frame(X.lda.data))
        qda.test.lda.pred <- predict(qda.test.lda,newdata=as.data.frame(X.lda.test))
        qda.test.lda.perf <- table(y.test,qda.test.lda.pred$class)
        qda.test.lda.error <- (1 - sum(diag(qda.test.lda.perf))/X.test.dim[1])*100
        qda.app.lda.error <- lda.min
      } else if( i == 2){ # LDA 
        lda.test.lda.k5 <- lda(y.app~., data=as.data.frame(X.lda.data))
        lda.test.lda.pred.k5 <- predict(lda.test.lda.k5,newdata=as.data.frame(X.lda.test))
        lda.test.lda.perf.k5 <- table(y.test,lda.test.lda.pred.k5$class)
        lda.test.lda.error.k5 <- (1 - sum(diag(lda.test.lda.perf.k5))/X.test.dim[1])*100
        lda.app.lda.error.k5 <- lda.min
      } else if( i == 12){ # LDA 
        lda.test.lda <- lda(y.app~., data=as.data.frame(X.lda.data))
        lda.test.lda.pred <- predict(lda.test.lda,newdata=as.data.frame(X.lda.test))
        lda.test.lda.perf <- table(y.test,lda.test.lda.pred$class)
        lda.test.lda.error <- (1 - sum(diag(lda.test.lda.perf))/X.test.dim[1])*100
        lda.app.lda.error <- lda.min
      } else if( i == 4) { # LogReg
        logReg.test.lda.res.k5<-c(rep(0,X.test.dim[1]))
        logReg.test.lda.k5 <- glmnet(X.lda.data ,y=y.app,family="multinomial")
        logReg.test.lda.pred.k5 <- predict(logReg.test.lda.k5,newx=X.lda.test,type="response",s=logReg.test.lda.k5$lambda.min)
        
        for (h in 1:X.test.dim[1]) {
          logReg.test.lda.res.k5[h] <-which.max(logReg.test.lda.pred.k5[h,1:6,dim(logReg.test.lda.pred.k5)[3]-1])
        }
        logReg.test.lda.perf.k5 <- table(y.test,logReg.test.lda.res.k5)
        logReg.test.lda.error.k5 <-(1 - sum(diag(logReg.test.lda.perf.k5))/X.test.dim[1])*100
        logReg.app.lda.error.k5 <- lda.min
      } else if( i == 14) { # LogReg
        logReg.test.lda.res<-c(rep(0,X.test.dim[1]))
        logReg.test.lda <- glmnet(X.lda.data ,y=y.app,family="multinomial")
        logReg.test.lda.pred <- predict(logReg.test.lda,newx=X.lda.test,type="response",s=logReg.test.lda$lambda.min)
        
        for (h in 1:X.test.dim[1]) {
          logReg.test.lda.res[h] <-which.max(logReg.test.lda.pred[h,1:6,dim(logReg.test.lda.pred)[3]-1])
        }
        logReg.test.lda.perf <- table(y.test,logReg.test.lda.res)
        logReg.test.lda.error <-(1 - sum(diag(logReg.test.lda.perf))/X.test.dim[1])*100
        logReg.app.lda.error <- lda.min
      } else if( i == 5) { # naive bayesien
        nb.test.lda.k5 <- naiveBayes(factor(y.app)~., data=as.data.frame(X.lda.data))
        nb.test.lda.pred.k5 <- predict(nb.test.lda.k5,newdata=as.data.frame(X.lda.test))
        nb.test.lda.perf.k5 <- table(factor(y.test),nb.test.lda.pred.k5)
        nb.test.lda.error.k5 <- (1 - sum(diag(nb.test.lda.perf.k5))/X.test.dim[1])*100
        nb.app.lda.error.k5 <- lda.min
      } else if( i == 15) { # naive bayesien
        nb.test.lda <- naiveBayes(factor(y.app)~., data=as.data.frame(X.lda.data))
        nb.test.lda.pred <- predict(nb.test.lda,newdata=as.data.frame(X.lda.test))
        nb.test.lda.perf <- table(factor(y.test),nb.test.lda.pred)
        nb.test.lda.error <- (1 - sum(diag(nb.test.lda.perf))/X.test.dim[1])*100
        nb.app.lda.error <- lda.min
      } else if( i == 6) { # SVM
        svm.test.lda.k5 <- svm(X.lda.data,y.app,type="C-classification")
        svm.test.lda.pred.k5 <- predict(svm.test.lda.k5, X.lda.test)
        svm.test.lda.perf.k5 <- table(factor(y.test),svm.test.lda.pred.k5)
        svm.test.lda.error.k5 <- (1 - sum(diag(svm.test.lda.perf.k5))/X.test.dim[1])*100
        svm.app.lda.error.k5 <- lda.min
      } else if( i == 16) { # SVM
        svm.test.lda <- svm(X.lda.data,y.app,type="C-classification")
        svm.test.lda.pred <- predict(svm.test.lda, X.lda.test)
        svm.test.lda.perf <- table(factor(y.test),svm.test.lda.pred)
        svm.test.lda.error <- (1 - sum(diag(svm.test.lda.perf))/X.test.dim[1])*100
        svm.app.lda.error <- lda.min
      }else if( i == 7) { # SVM Tune
        svm.tune.test.lda.k5 <- tune(svm, train.y = factor(y.app),  train.x = X.lda.data, ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))
        svm.tune.test.lda.pred.k5 <- predict(svm.tune.test.lda.k5$best.model , X.lda.test)
        svm.tune.test.lda.perf.k5 <- table(factor(y.test),svm.tune.test.lda.pred.k5)
        svm.tune.test.lda.error.k5 <- (1 - sum(diag(svm.tune.test.lda.perf.k5))/X.test.dim[1])*100
        svm.tune.app.lda.error.k5 <- lda.min
      } else if( i == 17) { # SVM Tune
        svm.tune.test.lda <- tune(svm, train.y = factor(y.app),  train.x = X.lda.data, ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))
        svm.tune.test.lda.pred <- predict(svm.tune.test.lda$best.model , X.lda.test)
        svm.tune.test.lda.perf <- table(factor(y.test),svm.tune.test.lda.pred)
        svm.tune.test.lda.error <- (1 - sum(diag(svm.tune.test.lda.perf))/X.test.dim[1])*100
        svm.tune.app.lda.error <- lda.min
      }else if( i == 8) { # Tree
        tree.test.lda.k5 <- tree(factor(y.app)~., data=as.data.frame(X.lda.data))
        tree.test.lda.pred.k5 <- predict(tree.test.lda.k5,newdata=as.data.frame(X.lda.test), type="class")
        tree.test.lda.perf.k5 <- table(y.test,tree.test.lda.pred.k5)
        tree.test.lda.error.k5 <- (1 - sum(diag(tree.test.lda.perf.k5))/X.test.dim[1])*100
        tree.app.lda.error.k5 <- lda.min
      } else if( i == 18) { # Tree
        tree.test.lda <- tree(factor(y.app)~., data=as.data.frame(X.lda.data))
        tree.test.lda.pred <- predict(tree.test.lda,newdata=as.data.frame(X.lda.test), type="class")
        tree.test.lda.perf <- table(y.test,tree.test.lda.pred)
        tree.test.lda.error <- (1 - sum(diag(tree.test.lda.perf))/X.test.dim[1])*100
        tree.app.lda.error <- lda.min
      } else if( i == 10) { # Neural network
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
          nn.app.lda.error.k5 <- lda.min 
      } else if( i == 20) { # Neural network
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
        nn.app.lda.error <- lda.min
      }
    }
  } else {
    print("forward")
    ind =  which(forward == min(forward), arr.ind = TRUE)
    print(ind)
  }
}

# Observation de nos meilleurs modeles maintenant qu'ils ont ete calcule

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


