# LOAD PACKAGES 
# =============
rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/data/data_expressions.RData")
library(MASS)
library(leaps)
library(class)


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
# KNN Analysis
# ===
K <- 5 # Nombre de sections dans notre ensemble d'apprentissage
folds <- sample(1:K,X.app.dim[1] ,replace=TRUE)

knn.acpF.error <- rep(0,40)
knn.acp1.error <- rep(0,40)
knn.acp2.error <- rep(0,40)
knn.lda.error <- rep(0,40)
knn.forward.error <- rep(0,40)

y.app <- factor(y.app)

for(i in 1:K){
  numberTest <- dim(X.acp.transform1[folds==i,])[1]

  for(k in 1:40){
    # ACP Full
    knn.acpF <- knn(as.data.frame(X.acp.full[folds!=i,]), as.data.frame(X.acp.full[folds==i,]), y.app[folds!=i],k=k)
    knn.acpF.perf <- table(y.app[folds==i], knn.acpF)
    knn.acpF.error[k] <-knn.acpF.error[k]  + 1 - sum(diag(knn.acpF.perf))/numberTest
    
    # ACP1 
    knn.acp1 <- knn(as.data.frame(X.acp.transform1[folds!=i,]), as.data.frame(X.acp.transform1[folds==i,]), y.app[folds!=i],k=k)
    knn.acp1.perf <- table(y.app[folds==i], knn.acp1)
    knn.acp1.error[k] <-knn.acp1.error[k]  + 1 - sum(diag(knn.acp1.perf))/numberTest
    
    # ACP2
    knn.acp2 <- knn(as.data.frame(X.acp.transform2[folds!=i,]), as.data.frame(X.acp.transform2[folds==i,]), y.app[folds!=i],k=k)
    knn.acp2.perf <- table(y.app[folds==i], knn.acp2)
    knn.acp2.error[k] <-knn.acp2.error[k]  + 1 - sum(diag(knn.acp2.perf))/numberTest
    
    # FDA
    knn.lda <- knn(as.data.frame(X.lda.transform[folds!=i,]), as.data.frame(X.lda.transform[folds==i,]), y.app[folds!=i],k=k)
    knn.lda.perf <- table(y.app[folds==i], knn.lda)
    knn.lda.error[k] <-knn.lda.error[k]  + 1 - sum(diag(knn.lda.perf))/numberTest
    
    # Forward
    knn.forward <- knn(as.data.frame(X.forward[folds!=i,]), as.data.frame(X.forward[folds==i,]), y.app[folds!=i],k=k)
    knn.forward.perf <- table(y.app[folds==i], knn.forward)
    knn.forward.error[k] <-knn.forward.error[k]  + 1 - sum(diag(knn.forward.perf))/numberTest
  }
}

# Diviser le taux d'erreur par le nombre de K 
knn.acpF.error <-(knn.acpF.error/K)*100
knn.acp1.error <-(knn.acp1.error/K)*100
knn.acp2.error <-(knn.acp2.error/K)*100
knn.lda.error <-(knn.lda.error/K)*100
knn.forward.error <-(knn.forward.error/K)*100

plot(knn.acpF.error, main="ACP FULL", type="l")
plot(knn.acp1.error, main="ACP 1", type="l")
plot(knn.acp2.error, main="ACP 2", type="l")
plot(knn.lda.error, main="FDA", type="l")
plot(knn.forward.error, main="Forward", type="l")

print(min(knn.acpF.error))
print(min(knn.acp1.error))
print(min(knn.acp2.error))
print(min(knn.lda.error))
print(min(knn.forward.error))



