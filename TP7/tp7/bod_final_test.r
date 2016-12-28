# LOAD PACKAGES 
# =============
rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/end2.rData")
library(MASS)
library(leaps)
library(glmnet)
library(class)
library(e1071)
library(tree)
library(randomForest)

vectorTree <- seq(100,1500,100)
methodArray <- c("qda.acp.error","qda.lda.error","qda.forward.error",
                 "lda.acp.error","lda.lda.error","lda.forward.error",
                 "knn.acp.error","knn.lda.error","knn.forward.error",
                 "logReg.acp.error","logReg.lda.error","logReg.forward.error",
                 "nb.acp.error","nb.lda.error","nb.forward.error",
                 "svm.acp.error","svm.lda.error","svm.forward.error",
                 "svm.tune.acp.error","svm.tune.lda.error","svm.tune.forward.error",
                 "tree.acp.error","tree.lda.error","tree.forward.error",
                 "rf.acp.error","rf.lda.error","rf.forward.error"
                )

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
      }
    }
  } else {
    print("forward")
    ind =  which(forward == min(forward), arr.ind = TRUE)
    print(ind)
  }
}
