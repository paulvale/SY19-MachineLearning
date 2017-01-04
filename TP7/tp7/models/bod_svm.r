# LOAD PACKAGES 
# =============
rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/data/data_expressions.RData")
library(e1071)
library(MASS)
library(leaps)


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

# ====
# Reduction de la dimensionnalite des variables
# ===
# 3 techniques de reduction de dimensionnalite a essaye :
# - FDA
# - ACP
# - Forward/Backward Selection

# 1) FDA
X.lda <- lda(y.app~., data=as.data.frame(X.app))
X.lda.prop = cumsum(X.lda$svd^2/sum(X.lda$svd^2))

X.lda.transform <- X.app%*%X.lda$scaling

# 2) ACP
X.acp <- prcomp(X.app)
acp.sum <- cumsum(100 * X.acp$sdev^2 / sum(X.acp$sdev^2))

X.acp.full <- X.acp$x
X.acp.transform1 <- X.acp$x[,which(acp.sum<75)]
X.acp.transform2 <- X.acp$x[,which(acp.sum<85)]

# 3) Forward/Backward Selection
reg.fit<-regsubsets(y.app~.,data=as.data.frame(X.app),method="forward",nvmax=50)

tmp <- which(reg.fit$rss > 50)
ind <- tmp[length(tmp)]

X.forward <- X.app[,which(reg.fit$vorder < ind)]

## === 
## Transformation aussi de nos donnees de test dans le meme referentiel pour toutes 
## ===

# 1) FDA
X.lda.transform.test <- X.test%*%X.lda$scaling

# 2) ACP
X.acp.test <- X.test %*% X.acp$rotation

X.acp.test.full <- X.acp.test
X.acp.test.transform1 <- X.acp.test[,which(acp.sum<75)]
X.acp.test.transform2 <- X.acp.test[,which(acp.sum<85)]

# 3) Forward/Backward Selection
X.forward.test <- X.test[,which(reg.fit$vorder < ind)]

# ===
# SVM 
# ===
# models
svm.acpF <- svm(X.acp.full,y.app,type="C-classification")
svm.acp1 <- svm(X.acp.transform1,y.app,type="C-classification")
svm.acp2 <- svm(X.acp.transform2,y.app,type="C-classification")
svm.lda <- svm(X.lda.transform,y.app, type="C-classification")
svm.forward <- svm(X.forward,y.app, type="C-classification")

# prediction
svm.acpF.prediction <- predict(svm.acpF, X.acp.test.full)
svm.acp1.prediction <- predict(svm.acp1, X.acp.test.transform1)
svm.acp2.prediction <- predict(svm.acp2, X.acp.test.transform2)
svm.lda.prediction <- predict(svm.lda, X.lda.transform.test)
svm.forward.prediction <- predict(svm.forward, X.forward.test)

# matrice de confusion
svm.acpF.confusion <- table(y.test, svm.acpF.prediction)
svm.acp1.confusion <- table(y.test, svm.acp1.prediction)
svm.acp2.confusion <- table(y.test, svm.acp2.prediction)
svm.lda.confusion <- table(y.test, svm.lda.prediction)
svm.forward.confusion <- table(y.test, svm.forward.prediction)

# perform a grid search
svm.acpF.tuneResult <- tune(svm, train.y = factor(y.app),  train.x = X.acp.full, ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))
svm.acp1.tuneResult <- tune(svm, train.y = factor(y.app),  train.x = X.acp.transform1,ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))
svm.acp2.tuneResult <- tune(svm, train.y = factor(y.app),  train.x = X.acp.transform2, ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))
svm.lda.tuneResult <- tune(svm, train.y = factor(y.app),  train.x = X.lda.transform, ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))
svm.forward.tuneResult <- tune(svm, train.y = factor(y.app),  train.x = X.forward, ranges = list(cost=10^(-1:2), gamma=c(.0005, .005, .05, .5, 1,2)))

# prediction after tune
svm.acpF.tune.prediction <- predict(svm.acpF.tuneResult$best.model, X.acp.test.full)
svm.acp1.tune.prediction <- predict(svm.acp1.tuneResult$best.model, X.acp.test.transform1)
svm.acp2.tune.prediction <- predict(svm.acp2.tuneResult$best.model, X.acp.test.transform2)
svm.lda.tune.prediction <- predict(svm.lda.tuneResult$best.model, X.lda.transform.test)
svm.forward.tune.prediction <- predict(svm.forward.tuneResult$best.model, X.forward.test)

# matrice de confusion
svm.acpF.tune.confusion <- table(factor(y.test), svm.acpF.tune.prediction)
svm.acp1.tune.confusion <- table(factor(y.test), svm.acp1.tune.prediction)
svm.acp2.tune.confusion <- table(factor(y.test), svm.acp2.tune.prediction)
svm.lda.tune.confusion <- table(factor(y.test), svm.lda.tune.prediction)
svm.forward.tune.confusion <- table(factor(y.test), svm.forward.tune.prediction)

# Erreur pour tout 
# avec tune
svm.acpF.tune.error <- 1 - sum(diag(svm.acpF.tune.confusion))/length(y.test)
svm.acp1.tune.error <- 1 - sum(diag(svm.acp1.tune.confusion))/length(y.test)
svm.acp2.tune.error <- 1 - sum(diag(svm.acp2.tune.confusion))/length(y.test)
svm.lda.tune.error <- 1 - sum(diag(svm.lda.tune.confusion))/length(y.test)
svm.forward.tune.error <- 1 - sum(diag(svm.forward.tune.confusion))/length(y.test)

# sans tune
svm.acpF.error <- 1 - sum(diag(svm.acpF.confusion))/length(y.test)
svm.acp1.error <- 1 - sum(diag(svm.acp1.confusion))/length(y.test)
svm.acp2.error <- 1 - sum(diag(svm.acp2.confusion))/length(y.test)
svm.lda.error <- 1 - sum(diag(svm.lda.confusion))/length(y.test)
svm.forward.error <- 1 - sum(diag(svm.forward.confusion))/length(y.test)

print(svm.acpF.error)
print(svm.acp1.error) 
print(svm.acp2.error) 
print(svm.lda.error) 
print(svm.forward.error) 

print(svm.acpF.error - svm.acpF.tune.error) 
print(svm.acp1.error - svm.acp1.tune.error)
print(svm.acp2.error - svm.acp2.tune.error)
print(svm.lda.error - svm.lda.tune.error) 
print(svm.forward.error - svm.forward.tune.error)



 

 

