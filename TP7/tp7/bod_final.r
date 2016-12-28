# LOAD PACKAGES 
# =============
rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/data/data_expressions.RData")
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



# declaration de fonciton utile pour le réseau de neurones
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

X.dim <- dim(X)

# reduction des variables dont les colonnes sont toujours egales a 0 
tab<- rep(0,4200)
for(i in 1:4200){
  if(X[1,i]==0){
    tab[i]<-1
  }
}
X.reduct <-X[,which(tab==0)]
X.reduct.dim <- dim(X.reduct)
firstReduct <- X.dim[2] - X.reduct.dim[2]

# ====
# Division de X en app et test
# ====
n <- round(0.8*36)
indice <- sample(1:36, n)

X.test <- NULL
y.test <- NULL

X.app <- NULL
y.app <- NULL

for(i in 1:6){
  value <-  X.reduct[which(y==i),]
  value.y <- y[which(y==i)]
  
  app <- value[indice,]
  test <- value[-indice,]
  label.app <- value.y[indice]
  label.test <- value.y[-indice]
  
  X.test <- rbind(X.test, test)
  X.app <- rbind(X.app, app)
  y.app <- c(y.app,label.app)
  y.test <- c(y.test, label.test)
}

X.test.dim <- dim(X.test)
X.app.dim <- dim(X.app)

K <- 5 # Nombre de sections dans notre ensemble d'apprentissage
numberKnn <- 40
vectorTree <- seq(100,1500,100)
folds <- sample(1:K,X.app.dim[1] ,replace=TRUE)

tree.acp.error <- rep(0,159)
tree.lda.error <- 0
tree.forward.error <- rep(0, 159)

svm.acp.error <- rep(0,159)
svm.lda.error <- 0
svm.forward.error <- rep(0, 159)

svm.tune.acp.error <- rep(0,159)
svm.tune.lda.error <- 0
svm.tune.forward.error <- rep(0, 159)

nb.acp.error <- rep(0,159)
nb.lda.error <- 0
nb.forward.error <- rep(0,159)

logReg.acp.error <- rep(0,159)
logReg.lda.error <- 0
logReg.forward.error <- rep(0, 159)

lda.acp.error <- rep(0,159)
lda.lda.error <- 0
lda.forward.error <- rep(0, 159)

qda.acp.error <- rep(0,5)
qda.lda.error <- 0
qda.forward.error <- rep(0,5)

knn.acp.error <- matrix(0,nrow=numberKnn, ncol=159)
knn.lda.error <- rep(0,numberKnn)
knn.forward.error <- matrix(0,nrow=numberKnn, ncol=159)

rf.acp.error <- matrix(0,nrow=length(vectorTree), ncol=159)
rf.lda.error <- rep(0,length(vectorTree))
rf.forward.error <- matrix(0,nrow=length(vectorTree), ncol=159)

nn.acp.error <- rep(0,159)
nn.lda.error <- 0
nn.forward.error <- rep(0, 159)

depart <- Sys.time()

for(i in 1:K){
  print("===")
  print(i)
  print("===")
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
  #plot(reg.fit)
  X.forward.data <- X.acp.data[,which(reg.fit$vorder == 1)]
  X.forward.data <- as.data.frame(X.forward.data)
  names(X.forward.data)[1] <- colnames(X.acp.data)[which(reg.fit$vorder == 1)]

  numberTest <- dim(X.lda.data[folds==i,])[1]
  y.app.factor <- factor(y.app[folds!=i])
  y.test.factor <- factor(y.app[folds==i])
  levels(y.test.factor) <- levels(y.app.factor)
  
  # ===
  # ACP
  # ===
  print("ACP")
  for(j in 2:160){
    print(j)
    # LDA
    lda.acp <- lda(y.app[folds!=i]~., data=as.data.frame(X.acp.data[folds!=i,1:j]))
    lda.acp.pred <- predict(lda.acp,newdata=as.data.frame(X.acp.data[folds==i,1:j]))
    lda.acp.perf <- table(y.app[folds==i],lda.acp.pred$class)
    lda.acp.error[j-1] <-lda.acp.error[j-1] + 1 - sum(diag(lda.acp.perf))/numberTest
    
    # KNN
    for(number in 1:numberKnn){
      # ACP Full
      knn.acp <- knn(as.data.frame(X.acp.data[folds!=i,1:j]), as.data.frame(X.acp.data[folds==i,1:j]), y.app[folds!=i],k=number)
      knn.acp.perf <- table(y.app[folds==i], knn.acp)
      knn.acp.error[number,j-1] <-knn.acp.error[number,j-1]  + 1 - sum(diag(knn.acp.perf))/numberTest
    }
    
    # LogReg
    logReg.acp.res<-c(rep(0,numberTest))
    logReg.acp <- glmnet(X.acp.data[folds!=i,1:j] ,y=y.app[folds!=i],family="multinomial")
    logReg.acp.pred <- predict(logReg.acp,newx=X.acp.data[folds==i,1:j],type="response",s=logReg.acp$lambda.min)
    
    for (h in 1:numberTest) {
      logReg.acp.res[h] <-which.max(logReg.acp.pred[h,1:6,dim(logReg.acp.pred)[3]-1])
    }
    
    logReg.acp.perf <- table(y.app[folds==i],logReg.acp.res)
    logReg.acp.error[j-1] <-logReg.acp.error[j-1] + 1 - sum(diag(logReg.acp.perf))/numberTest
    
    
    # Random Forest
    for(tree in vectorTree) {
      rf.acp <- randomForest(y.app.factor~., data=as.data.frame(X.acp.data[folds!=i,1:j]), xtest=as.data.frame(X.acp.data[folds==i,1:j]), ytest=y.test.factor, ntree=tree)
      rf.acp.error[tree/100, j-1] <- rf.acp.error[tree/100, j-1] + 1 - (sum(diag(rf.acp$test$confusion)/numberTest))
    }
        
    # Naive Baeysien
    nb.acp <- naiveBayes(factor(y.app[folds!=i])~., data=as.data.frame(X.acp.data[folds!=i,1:j]))
    nb.acp.pred <- predict(nb.acp,newdata=as.data.frame(X.acp.data[folds==i,1:j]))
    nb.acp.perf <- table(factor(y.app[folds==i]),nb.acp.pred)
    nb.acp.error[j-1] <-nb.acp.error[j-1] + 1 - sum(diag(nb.acp.perf))/numberTest
    
    # SVM
    svm.acp <- svm(X.acp.data[folds!=i,1:j],y.app[folds!=i],type="C-classification")
    svm.acp.pred <- predict(svm.acp, X.acp.data[folds==i,1:j])
    svm.acp.perf <- table(factor(y.app[folds==i]),svm.acp.pred)
    svm.acp.error[j-1] <-svm.acp.error[j-1] + 1 - sum(diag(svm.acp.perf))/numberTest
    
    # SVM Tune 
    svm.tune.acp <- tune(svm, train.y = factor(y.app[folds!=i]),  train.x = X.acp.data[folds!=i,1:j], ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))
    svm.tune.acp.pred <- predict(svm.tune.acp$best.model, X.acp.data[folds==i,1:j])
    svm.tune.acp.perf <- table(factor(y.app[folds==i]),svm.tune.acp.pred)
    svm.tune.acp.error[j-1] <-svm.tune.acp.error[j-1] + 1 - sum(diag(svm.tune.acp.perf))/numberTest
    
    # Tree
    tree.acp <- tree(factor(y.app[folds!=i])~., data=as.data.frame(X.acp.data[folds!=i,1:j]))
    tree.acp.pred <- predict(tree.acp,as.data.frame(X.acp.data[folds==i,1:j]), type="class")
    tree.acp.perf <- table(y.app[folds==i],tree.acp.pred)
    tree.acp.error[j-1] <-tree.acp.error[j-1] + 1 - sum(diag(tree.acp.perf))/numberTest

    # Neural Network
    data.train = X.acp.data[folds!=i,1:j]
    data.test = X.acp.data[folds==i,1:j]
    y.train = y.app[folds!=i]
    y.testfold = y.app[folds==i]
    ordre = c(1:dim(data.train)[2])
    for(kk in 1:6){
        response <- rep(0,length(y.train))
        for(j in 1:length(y.train)){
          if(y.train[j]==k){
            response[j]=1
          }
        }
        newDataSet <- data.frame(data.train,response)
        formule = getFormulas(colnames(newDataSet), ordre,"response")
        if(kk==1)
        {
          neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if( kk==2)
        {
          neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==3){
          neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==4){
          neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==5){
          neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==6){
          neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
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

      for(j in 1:dim(data.test)[1]){
        index <- which.max(c(c1$net.result[j],c2$net.result[j],c3$net.result[j],c4$net.result[j],c5$net.result[j],c6$net.result[j]))
        neuralnet.result[j] <- index
      }
      neuralnet.perf <- table(neuralnet.result,y.testfold)
      nn.acp.error[j-1] <- 1-sum(diag(neuralnet.perf))/length(y.testfold)
  }
  
  # ===
  # Forward
  # ===
  print("Forward")
  for(j in 2:160){
    print(j)
    X.forward.data[[colnames(X.acp.data)[which(reg.fit$vorder == j)]]] <- X.acp.data[,which(reg.fit$vorder == j)]

    # LDA
    lda.forward <- lda(y.app[folds!=i]~., data=as.data.frame(X.forward.data[folds!=i,1:j]))
    lda.forward.pred <- predict(lda.forward,newdata=as.data.frame(X.forward.data[folds==i,1:j]))
    lda.forward.perf <- table(y.app[folds==i],lda.forward.pred$class)
    lda.forward.error[j-1] <-lda.forward.error[j-1] + 1 - sum(diag(lda.forward.perf))/numberTest
    
    # KNN
    for(number in 1:numberKnn){
    # ACP Full
      knn.forward <- knn(as.data.frame(X.forward.data[folds!=i,1:j]), as.data.frame(X.forward.data[folds==i,1:j]), y.app[folds!=i],k=number)
      knn.forward.perf <- table(y.app[folds==i], knn.forward)
      knn.forward.error[number,j-1] <-knn.forward.error[number,j-1]  + 1 - sum(diag(knn.forward.perf))/numberTest
    }
    
    # LogReg
    logReg.forward.res<-c(rep(0,numberTest))
    logReg.forward <- glmnet(as.matrix(X.forward.data[folds!=i,1:j]) ,y=y.app[folds!=i],family="multinomial")
    logReg.forward.pred <- predict(logReg.forward,newx=as.matrix(X.forward.data[folds==i,1:j]),type="response",s=logReg.forward$lambda.min)
    for (h in 1:numberTest) {
      logReg.forward.res[h] <-which.max(logReg.forward.pred[h,1:6,dim(logReg.forward.pred)[3] -1])
    }
    
    logReg.forward.perf <- table(y.app[folds==i],logReg.forward.res)
    logReg.forward.error[j-1] <-logReg.forward.error[j-1] + 1 - sum(diag(logReg.forward.perf))/numberTest
    
    # Random Forest
    for(tree in vectorTree) {
      rf.forward <- randomForest(y.app.factor~., data=as.data.frame(X.forward.data[folds!=i,1:j]), xtest=as.data.frame(X.forward.data[folds==i,1:j]), ytest=y.test.factor, ntree=tree)
      rf.forward.error[tree/100, j-1] <- rf.forward.error[tree/100, j-1] + 1 - (sum(diag(rf.forward$test$confusion)/numberTest))
    }
    
    # Naive Baeysien
    nb.forward <- naiveBayes(factor(y.app[folds!=i])~., data=as.data.frame(X.forward.data[folds!=i,1:j]))
    nb.forward.pred <- predict(nb.forward,newdata=as.data.frame(X.forward.data[folds==i,1:j]))
    nb.forward.perf <- table(factor(y.app[folds==i]),nb.forward.pred)
    nb.forward.error[j-1] <-nb.forward.error[j-1] + 1 - sum(diag(nb.forward.perf))/numberTest
    
    # SVM
    svm.forward <- svm(X.forward.data[folds!=i,1:j],y.app[folds!=i],type="C-classification")
    svm.forward.pred <- predict(svm.forward, X.forward.data[folds==i,1:j])
    svm.forward.perf <- table(factor(y.app[folds==i]),svm.forward.pred)
    svm.forward.error[j-1] <-svm.forward.error[j-1] + 1 - sum(diag(svm.forward.perf))/numberTest
    
    # SVM Tune 
    svm.tune.forward <- tune(svm, train.y = factor(y.app[folds!=i]),  train.x = X.forward.data[folds!=i,1:j], ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))
    svm.tune.forward.pred <- predict(svm.tune.forward$best.model, X.forward.data[folds==i,1:j])
    svm.tune.forward.perf <- table(factor(y.app[folds==i]),svm.tune.forward.pred)
    svm.tune.forward.error[j-1] <-svm.tune.forward.error[j-1] + 1 - sum(diag(svm.tune.forward.perf))/numberTest
    
    # Tree
    tree.forward <- tree(factor(y.app[folds!=i])~., data=as.data.frame(X.forward.data[folds!=i,1:j]))
    tree.forward.pred <- predict(tree.forward,as.data.frame(X.forward.data[folds==i,1:j]), type="class")
    tree.forward.perf <- table(y.app[folds==i],tree.forward.pred)
    tree.forward.error[j-1] <-tree.forward.error[j-1] + 1 - sum(diag(tree.forward.perf))/numberTest

    # Neural Network
    data.train = X.forward.data[folds!=i,1:j]
    data.test = X.forward.data[folds==i,1:j]
    y.train = y.app[folds!=i]
    y.testfold = y.app[folds==i]
    ordre = c(1:dim(data.train)[2])
    for(kk in 1:6){
        response <- rep(0,length(y.train))
        for(j in 1:length(y.train)){
          if(y.train[j]==k){
            response[j]=1
          }
        }
        newDataSet <- data.frame(data.train,response)
        formule = getFormulas(colnames(newDataSet), ordre,"response")
        if(kk==1)
        {
          neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if( kk==2)
        {
          neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==3){
          neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==4){
          neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==5){
          neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==6){
          neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
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

      for(j in 1:dim(data.test)[1]){
        index <- which.max(c(c1$net.result[j],c2$net.result[j],c3$net.result[j],c4$net.result[j],c5$net.result[j],c6$net.result[j]))
        neuralnet.result[j] <- index
      }
      neuralnet.perf <- table(neuralnet.result,y.testfold)
      nn.forward.error[j-1] <- 1-sum(diag(neuralnet.perf))/length(y.testfold)
  }
  
  
  # ===
  # FDA
  # ===
  print("FDA")
  # Random Forest
  for(tree in vectorTree) {
    rf.lda <- randomForest(y.app.factor~., data=as.data.frame(X.lda.data[folds!=i,]), xtest=as.data.frame(X.lda.data[folds==i,]), ytest=y.test.factor, ntree=tree)
    rf.lda.error[tree/100] <- rf.lda.error[tree/100] + 1 - (sum(diag(rf.lda$test$confusion)/numberTest))
  }
  
  # Tree
  tree.lda <- tree(factor(y.app[folds!=i])~., data=as.data.frame(X.lda.data[folds!=i,]))
  tree.lda.pred <- predict(tree.lda,newdata=as.data.frame(X.lda.data[folds==i,]), type="class")
  tree.lda.perf <- table(y.app[folds==i],tree.lda.pred)
  tree.lda.error <-tree.lda.error + 1 - sum(diag(tree.lda.perf))/numberTest
  
  # SVM
  svm.lda <- svm(X.lda.data[folds!=i,],y.app[folds!=i],type="C-classification")
  svm.lda.pred <- predict(svm.lda, X.lda.data[folds==i,])
  svm.lda.perf <- table(factor(y.app[folds==i]),svm.lda.pred)
  svm.lda.error <-svm.lda.error + 1 - sum(diag(svm.lda.perf))/numberTest
  
  # SVM Tune
  svm.tune.lda <- tune(svm, train.y = factor(y.app[folds!=i]),  train.x = X.lda.data[folds!=i,], ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))
  svm.tune.lda.pred <- predict(svm.tune.lda$best.model , X.lda.data[folds==i,])
  svm.tune.lda.perf <- table(factor(y.app[folds==i]),svm.tune.lda.pred)
  svm.tune.lda.error <-svm.tune.lda.error + 1 - sum(diag(svm.tune.lda.perf))/numberTest
  
  # Naive Bayesien
  nb.lda <- naiveBayes(factor(y.app[folds!=i])~., data=as.data.frame(X.lda.data[folds!=i,]))
  nb.lda.pred <- predict(nb.lda,newdata=as.data.frame(X.lda.data[folds==i,]))
  nb.lda.perf <- table(factor(y.app[folds==i]),nb.lda.pred)
  nb.lda.error <-nb.lda.error + 1 - sum(diag(nb.lda.perf))/numberTest
  
  # RegLog
  logReg.lda.res<-c(rep(0,numberTest))
  logReg.lda <- glmnet(X.lda.data[folds!=i,] ,y=y.app[folds!=i],family="multinomial")
  logReg.lda.pred <- predict(logReg.lda,newx=X.lda.data[folds==i,],type="response",s=logReg.lda$lambda.min)
  
  for (h in 1:numberTest) {
    logReg.lda.res[h] <-which.max(logReg.lda.pred[h,1:6,dim(logReg.lda.pred)[3] -1])
  }
  logReg.lda.perf <- table(y.app[folds==i],logReg.lda.res)
  logReg.lda.error <-logReg.lda.error + 1 - sum(diag(logReg.lda.perf))/numberTest
  
  # LDA
  lda.lda <- lda(y.app[folds!=i]~., data=as.data.frame(X.lda.data[folds!=i,]))
  lda.lda.pred <- predict(lda.lda,newdata=as.data.frame(X.lda.data[folds==i,]))
  lda.lda.perf <- table(y.app[folds==i],lda.lda.pred$class)
  lda.lda.error <-lda.lda.error + 1 - sum(diag(lda.lda.perf))/numberTest
  
  # KNN
  for(number in 1:numberKnn){
    knn.lda <- knn(as.data.frame(X.lda.data[folds!=i,]), as.data.frame(X.lda.data[folds==i,]), y.app[folds!=i],k=number)
    knn.lda.perf <- table(y.app[folds==i], knn.lda)
    knn.lda.error[number] <-knn.lda.error[number]  + 1 - sum(diag(knn.lda.perf))/numberTest 
  }
  
  # ===
  # Quadratic Discriminant Analysis
  # ===
  print("QDA")
  for(j in 2:6){
    # ACP
    qda.acp <- qda(y.app[folds!=i]~., data=as.data.frame(X.acp.data[folds!=i,1:j]))
    qda.acp.pred <- predict(qda.acp,newdata=as.data.frame(X.acp.data[folds==i,1:j]))
    qda.acp.perf <- table(y.app[folds==i],qda.acp.pred$class)
    qda.acp.error[j-1] <-qda.acp.error[j-1] + 1 - sum(diag(qda.acp.perf))/numberTest
    
    # Forward
    qda.forward <- qda(y.app[folds!=i]~., data=as.data.frame(X.forward.data[folds!=i,1:j]))
    qda.forward.pred <- predict(qda.forward,newdata=as.data.frame(X.forward.data[folds==i,1:j]))
    qda.forward.perf <- table(y.app[folds==i],qda.forward.pred$class)
    qda.forward.error[j-1] <-qda.forward.error[j-1] + 1 - sum(diag(qda.forward.perf))/numberTest
  }
  
  # FDA
  qda.lda <- qda(y.app[folds!=i]~., data=as.data.frame(X.lda.data[folds!=i,]))
  qda.lda.pred <- predict(qda.lda,newdata=as.data.frame(X.lda.data[folds==i,]))
  qda.lda.perf <- table(y.app[folds==i],qda.lda.pred$class)
  qda.lda.error <-qda.lda.error + 1 - sum(diag(qda.lda.perf))/numberTest

    # Neural Network
    data.train = X.lda.data[folds!=i,1:j]
    data.test = X.lda.data[folds==i,1:j]
    y.train = y.app[folds!=i]
    y.testfold = y.app[folds==i]
    ordre = c(1:dim(data.train)[2])
    for(kk in 1:6){
        response <- rep(0,length(y.train))
        for(j in 1:length(y.train)){
          if(y.train[j]==k){
            response[j]=1
          }
        }
        newDataSet <- data.frame(data.train,response)
        formule = getFormulas(colnames(newDataSet), ordre,"response")
        if(kk==1)
        {
          neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if( kk==2)
        {
          neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==3){
          neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==4){
          neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==5){
          neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)

        }
        else if(kk==6){
          neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
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
      for(j in 1:dim(data.test)[1]){
        index <- which.max(c(c1$net.result[j],c2$net.result[j],c3$net.result[j],c4$net.result[j],c5$net.result[j],c6$net.result[j]))
        neuralnet.result[j] <- index
      }
      neuralnet.perf <- table(neuralnet.result,y.testfold)
      nn.lda.error[j-1] <- 1-sum(diag(neuralnet.perf))/length(y.testfold)
}

# Diviser le taux d'erreur par le nombre de K 
qda.acp.error <-(qda.acp.error/K)*100
qda.lda.error <-(qda.lda.error/K)*100
qda.forward.error <-(qda.forward.error/K)*100

lda.acp.error <-(lda.acp.error/K)*100
lda.lda.error <-(lda.lda.error/K)*100
lda.forward.error <-(lda.forward.error/K)*100

knn.acp.error <- (knn.acp.error/K)*100
knn.lda.error <- (knn.lda.error/K)*100
knn.forward.error <- (knn.forward.error/K)*100

logReg.acp.error <- (logReg.acp.error/K)*100
logReg.lda.error <- (logReg.lda.error/K)*100
logReg.forward.error <- (logReg.forward.error/K)*100

nb.acp.error <- (nb.acp.error/K)*100
nb.lda.error <- (nb.lda.error/K)*100
nb.forward.error <- (nb.forward.error/K)*100

svm.acp.error <- (svm.acp.error/K)*100
svm.lda.error <- (svm.lda.error/K)*100
svm.forward.error <- (svm.forward.error/K)*100

svm.tune.acp.error <- (svm.tune.acp.error/K)*100
svm.tune.lda.error <- (svm.tune.lda.error/K)*100
svm.tune.forward.error <- (svm.tune.forward.error/K)*100

tree.acp.error <- (tree.acp.error/K)*100
tree.lda.error <- (tree.lda.error/K)*100
tree.forward.error <- (tree.forward.error/K)*100

rf.acp.error <- (rf.acp.error/K)*100
rf.lda.error <- (rf.lda.error/K)*100
rf.forward.error <- (rf.forward.error/K)*100

nn.acp.error <- (nn.acp.error/K)*100
nn.lda.error <- (nn.lda.error/K)*100
nn.forward.error <- (nn.forward.error/K)*100

print("QDA :")
print(min(qda.acp.error))
print(min(qda.lda.error))
print(min(qda.forward.error))

print("LDA :")
print(min(lda.acp.error))
print(min(lda.lda.error))
print(min(lda.forward.error))

print("KNN :")
print(min(knn.acp.error))
print(min(knn.lda.error))
print(min(knn.forward.error))

print("RegLog :")
print(min(logReg.acp.error))
print(min(logReg.lda.error))
print(min(logReg.forward.error))

print("Naive Bayesien :")
print(min(nb.acp.error))
print(min(nb.lda.error))
print(min(nb.forward.error))

print("SVM :")
print(min(svm.acp.error))
print(min(svm.lda.error))
print(min(svm.forward.error))

print("SVM Tune:")
print(min(svm.tune.acp.error))
print(min(svm.tune.lda.error))
print(min(svm.tune.forward.error))

print("Tree:")
print(min(tree.acp.error))
print(min(tree.lda.error))
print(min(tree.forward.error))

print("Random Forest :")
print(min(rf.acp.error))
print(min(rf.lda.error))
print(min(rf.forward.error))

print("Neural network :")
print(min(nn.acp.error))
print(min(nn.lda.error))
print(min(nn.forward.error))

final <- Sys.time()
print("temp final d'execution :")
print(final-depart)
