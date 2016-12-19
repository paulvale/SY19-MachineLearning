# LOAD PACKAGES 
# =============
rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/data/data_expressions.RData")
library(MASS)
library(leaps)
library(glmnet)


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

X.acp.full <- X.acp$x[,1:166]
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

X.acp.test.full <- X.acp.test[,1:166]
X.acp.test.transform1 <- X.acp.test[,which(acp.sum<75)]
X.acp.test.transform2 <- X.acp.test[,which(acp.sum<85)]

# 3) Forward/Backward Selection
X.forward.test <- X.test[,which(reg.fit$vorder < ind)]


# ===
# Linear Discriminant Analysis
# ===
K <- 5 # Nombre de sections dans notre ensemble d'apprentissage
folds <- sample(1:K,X.app.dim[1] ,replace=TRUE)
qda.acpF.error <- 0
qda.acp1.error <- 0
qda.acp2.error <- 0
qda.lda.error <- 0
qda.forward.error <- 0

for(i in 1:K){
  numberTest <- dim(X.acp.transform1[folds==i,])[1]
  print("i")

  # ACP Full
  qda.acpF <- qda(y.app[folds!=i]~., data=as.data.frame(X.acp.full[folds!=i,]))
  qda.acpF.pred <- predict(qda.acpF,newdata=as.data.frame(X.acp.full[folds==i,]))
  qda.acpF.perf <- table(y.app[folds==i],qda.acpF.pred$class)
  qda.acpF.error <-qda.acpF.error + 1 - sum(diag(qda.acpF.perf))/numberTest
  print("i")
  # ACP 1
  qda.acp1 <- qda(y.app[folds!=i]~., data=as.data.frame(X.acp.transform1[folds!=i,]))
  qda.acp1.pred <- predict(qda.acp1,newdata=as.data.frame(X.acp.transform1[folds==i,]))
  qda.acp1.perf <- table(y.app[folds==i],qda.acp1.pred$class)
  qda.acp1.error <-qda.acp1.error + 1 - sum(diag(qda.acp1.perf))/numberTest
  print("i")
  # ACP 2
  qda.acp2 <- qda(y.app[folds!=i]~., data=as.data.frame(X.acp.transform2[folds!=i,]))
  qda.acp2.pred <- predict(qda.acp1,newdata=as.data.frame(X.acp.transform2[folds==i,]))
  qda.acp2.perf <- table(y.app[folds==i],qda.acp2.pred$class)
  qda.acp2.error <-qda.acp2.error + 1 - sum(diag(qda.acp2.perf))/numberTest
  print("i")
  # FDA
  qda.lda <- qda(y.app[folds!=i]~., data=as.data.frame(X.lda.transform[folds!=i,]))
  qda.lda.pred <- predict(qda.lda,newdata=as.data.frame(X.lda.transform[folds==i,]))
  qda.lda.perf <- table(y.app[folds==i],qda.lda.pred$class)
  qda.lda.error <-qda.lda.error + 1 - sum(diag(qda.lda.perf))/numberTest
  print("i")
  # Forward
  qda.forward <- qda(y.app[folds!=i]~., data=as.data.frame(X.forward[folds!=i,]))
  qda.forward.pred <- predict(qda.forward,newdata=as.data.frame(X.forward[folds==i,]))
  qda.forward.perf <- table(y.app[folds==i],qda.forward.pred$class)
  qda.forward.error <-qda.forward.error + 1 - sum(diag(qda.forward.perf))/numberTest
}

# Diviser le taux d'erreur par le nombre de K 
qda.acpF.error <-(qda.acpF.error/K)*100
qda.acp1.error <-(qda.acp1.error/K)*100
qda.acp2.error <-(qda.acp2.error/K)*100
qda.lda.error <-(qda.lda.error/K)*100
qda.forward.error <-(qda.forward.error/K)*100

print(qda.acpF.error)
print(qda.acp1.error)
print(qda.acp2.error)
print(qda.lda.error)
print(qda.forward.error)


