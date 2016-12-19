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
# Naive Bayesien Analysis
# ===
K <- 5 # Nombre de sections dans notre ensemble d'apprentissage
folds <- sample(1:K,X.app.dim[1] ,replace=TRUE)
nb.acpF.error <- 0
nb.acp1.error <- 0
nb.acp2.error <- 0
nb.lda.error <- 0
nb.forward.error <- 0

# Attention
y.app <- factor(y.app)
y.test <- factor(y.test)

for(i in 1:K){
  numberTest <- dim(X.acp.transform1[folds==i,])[1]
  
  # ACP Full
  nb.acpF <- naiveBayes(y.app[folds!=i]~., data=as.data.frame(X.acp.full[folds!=i,]))
  nb.acpF.pred <- predict(nb.acpF,newdata=as.data.frame(X.acp.full[folds==i,]))
  nb.acpF.perf <- table(y.app[folds==i],nb.acpF.pred)
  nb.acpF.error <-nb.acpF.error + 1 - sum(diag(nb.acpF.perf))/numberTest
  
  # ACP 1
  nb.acp1 <- naiveBayes(y.app[folds!=i]~., data=as.data.frame(X.acp.transform1[folds!=i,]))
  nb.acp1.pred <- predict(nb.acp1,newdata=as.data.frame(X.acp.transform1[folds==i,]))
  nb.acp1.perf <- table(y.app[folds==i],nb.acp1.pred)
  nb.acp1.error <-nb.acp1.error + 1 - sum(diag(nb.acp1.perf))/numberTest
  
  # ACP 2
  nb.acp2 <- naiveBayes(y.app[folds!=i]~., data=as.data.frame(X.acp.transform2[folds!=i,]))
  nb.acp2.pred <- predict(nb.acp2,newdata=as.data.frame(X.acp.transform2[folds==i,]))
  nb.acp2.perf <- table(y.app[folds==i],nb.acp2.pred)
  nb.acp2.error <-nb.acp2.error + 1 - sum(diag(nb.acp2.perf))/numberTest
  
  # FDA
  nb.lda <- naiveBayes(y.app[folds!=i]~., data=as.data.frame(X.lda.transform[folds!=i,]))
  nb.lda.pred <- predict(nb.lda,newdata=as.data.frame(X.lda.transform[folds==i,]))
  nb.lda.perf <- table(y.app[folds==i],nb.lda.pred)
  nb.lda.error <-nb.lda.error + 1 - sum(diag(nb.lda.perf))/numberTest
  
  # Forward
  nb.forward <- naiveBayes(y.app[folds!=i]~., data=as.data.frame(X.forward[folds!=i,]))
  nb.forward.pred <- predict(nb.forward,newdata=as.data.frame(X.forward[folds==i,]))
  nb.forward.perf <- table(y.app[folds==i],nb.forward.pred)
  nb.forward.error <-nb.forward.error + 1 - sum(diag(nb.forward.perf))/numberTest
}

# Diviser le taux d'erreur par le nombre de K 
nb.acpF.error <-(nb.acpF.error/K)*100
nb.acp1.error <-(nb.acp1.error/K)*100
nb.acp2.error <-(nb.acp2.error/K)*100
nb.lda.error <-(nb.lda.error/K)*100
nb.forward.error <-(nb.forward.error/K)*100

print(nb.acpF.error)
print(nb.acp1.error)
print(nb.acp2.error)
print(nb.lda.error)
print(nb.forward.error)

