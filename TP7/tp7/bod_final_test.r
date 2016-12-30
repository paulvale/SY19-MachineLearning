# LOAD PACKAGES 
# =============
rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/endNN.rData")
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


vectorTree <- seq(100,1500,100)
methodArray <- c("qda.acp.error","qda.lda.error","qda.forward.error",
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
      print("knn Test Result :")
      ind =  which(lda == min(lda), arr.ind = TRUE) 
      knn.test.lda <- knn(as.data.frame(X.lda.data), as.data.frame(X.lda.test), y.app,k=ind[1])
      knn.test.lda.perf <- table(y.test, knn.test.lda)
      knn.test.lda.error <- (1 - sum(diag(knn.test.lda.perf))/X.test.dim[1])*100
      cat("app :", lda.min, "\n")
      cat("test :", knn.test.lda.error, "\n")
    } else if (i == 9){
      print("rf Test Result :")
      ind =  which(lda == min(lda), arr.ind = TRUE) 
      numberofTree <- vectorTree[ind]
      rf.test.lda <- randomForest(y.test.app.factor~., data=as.data.frame(X.lda.data), xtest=as.data.frame(X.lda.test), ytest=y.test.test.factor, ntree=numberofTree)
      rf.test.lda.error <- (1 - (sum(diag(rf.test.lda$test$confusion)/X.test.dim[1])))*100
      cat("app :", lda.min, "\n")
      cat("test :", rf.test.lda.error, "\n")
    } else {
      if( i == 1) { # QDA
        print("QDA Test Result :")
        qda.test.lda <- qda(y.app~., data=as.data.frame(X.lda.data))
        qda.test.lda.pred <- predict(qda.test.lda,newdata=as.data.frame(X.lda.test))
        qda.test.lda.perf <- table(y.test,qda.test.lda.pred$class)
        qda.test.lda.error <- (1 - sum(diag(qda.test.lda.perf))/X.test.dim[1])*100
        cat("app :", lda.min, "\n")
        cat("test :", qda.test.lda.error, "\n")
      } else if( i == 2){ # LDA 
        print("LDA Test Result :")
        lda.test.lda <- lda(y.app~., data=as.data.frame(X.lda.data))
        lda.test.lda.pred <- predict(lda.test.lda,newdata=as.data.frame(X.lda.test))
        lda.test.lda.perf <- table(y.test,lda.test.lda.pred$class)
        lda.test.lda.error <- (1 - sum(diag(lda.test.lda.perf))/X.test.dim[1])*100
        cat("app :", lda.min, "\n")
        cat("test :", lda.test.lda.error, "\n")
      } else if( i == 4) { # LogReg
        print("LogReg Test Result :")
        logReg.test.lda.res<-c(rep(0,X.test.dim[1]))
        logReg.test.lda <- glmnet(X.lda.data ,y=y.app,family="multinomial")
        logReg.test.lda.pred <- predict(logReg.test.lda,newx=X.lda.test,type="response",s=logReg.test.lda$lambda.min)
        
        for (h in 1:X.test.dim[1]) {
          logReg.test.lda.res[h] <-which.max(logReg.test.lda.pred[h,1:6,dim(logReg.test.lda.pred)[3]-1])
        }
        logReg.test.lda.perf <- table(y.test,logReg.test.lda.res)
        logReg.test.lda.error <-(1 - sum(diag(logReg.test.lda.perf))/X.test.dim[1])*100
        cat("app :", lda.min, "\n")
        cat("test :", logReg.test.lda.error, "\n")
      } else if( i == 5) { # naive bayesien
        print("NB Test Result :")
        nb.test.lda <- naiveBayes(factor(y.app)~., data=as.data.frame(X.lda.data))
        nb.test.lda.pred <- predict(nb.test.lda,newdata=as.data.frame(X.lda.test))
        nb.test.lda.perf <- table(factor(y.test),nb.test.lda.pred)
        nb.test.lda.error <- (1 - sum(diag(nb.test.lda.perf))/X.test.dim[1])*100
        cat("app :", lda.min, "\n")
        cat("test :", nb.test.lda.error, "\n")
      } else if( i == 6) { # SVM
        print("SVM Test Result :")
        svm.test.lda <- svm(X.lda.data,y.app,type="C-classification")
        svm.test.lda.pred <- predict(svm.test.lda, X.lda.test)
        svm.test.lda.perf <- table(factor(y.test),svm.test.lda.pred)
        svm.test.lda.error <- (1 - sum(diag(svm.test.lda.perf))/X.test.dim[1])*100
        cat("app :", lda.min, "\n")
        cat("test :", svm.test.lda.error, "\n")
      } else if( i == 7) { # SVM Tune
        print("SVM TUNE Test Result :")
        svm.tune.test.lda <- tune(svm, train.y = factor(y.app),  train.x = X.lda.data, ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))
        svm.tune.test.lda.pred <- predict(svm.tune.test.lda$best.model , X.lda.test)
        svm.tune.test.lda.perf <- table(factor(y.test),svm.tune.test.lda.pred)
        svm.tune.test.lda.error <- (1 - sum(diag(svm.tune.test.lda.perf))/X.test.dim[1])*100
        cat("app :", lda.min, "\n")
        cat("test :", svm.tune.test.lda.error, "\n")
      } else if( i == 8) { # Tree
        print("Tree Test Result :") 
        tree.test.lda <- tree(factor(y.app)~., data=as.data.frame(X.lda.data))
        tree.test.lda.pred <- predict(tree.test.lda,newdata=as.data.frame(X.lda.test), type="class")
        tree.test.lda.perf <- table(y.test,tree.test.lda.pred)
        tree.test.lda.error <- (1 - sum(diag(tree.test.lda.perf))/X.test.dim[1])*100
        cat("app :", lda.min, "\n")
        cat("test :", tree.test.lda.error, "\n")
      }else if( i == 10) { # Neural network
        print("Neural network Test Result :") 
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
          cat("app :", lda.min, "\n")
          cat("test :", nn.test.lda.error, "\n")
      }
    }
  } else {
    print("forward")
    ind =  which(forward == min(forward), arr.ind = TRUE)
    print(ind)
  }
}

# Observation de nos meilleurs modeles maintenant qu'ils ont ete calcule

