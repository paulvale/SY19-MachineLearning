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

# ===
# Quadratic Discriminant Analysis
# ===
K <- 5 # Nombre de sections dans notre ensemble d'apprentissage
folds <- sample(1:K,X.app.dim[1] ,replace=TRUE)
qda.acp.error <- rep(0,5) # 29 expressions face app 
qda.lda.error <- 0
qda.forward.error <- rep(0,5)

for(i in 1:K){
  print(i)
  # ====
  # Reduction de la dimensionnalite des variables
  # ===
  # 1) FDA
  X.lda <- lda(y.app~., data=as.data.frame(X.app))
  #X.lda.prop = cumsum(X.lda$svd^2/sum(X.lda$svd^2))
  
  X.lda.data <- X.app%*%X.lda$scaling
  
  # 2) ACP
  X.acp <- prcomp(X.app)
  #acp.sum <- cumsum(100 * X.acp$sdev^2 / sum(X.acp$sdev^2))
  
  X.acp.data <- X.acp$x
  
  # 3) Forward/Backward Selection
  reg.fit<-regsubsets(y.app~.,data=as.data.frame(X.app),method="forward")
  
  #tmp <- which(reg.fit$rss > 50)
  #ind <- tmp[length(tmp)]
  
  #X.forward <- X.app[,which(reg.fit$vorder < ind)]
  
  numberTest <- dim(X.lda.data[folds==i,])[1]

  
  for(j in 2:6){
    # ACP
    qda.acp <- qda(y.app[folds!=i]~., data=as.data.frame(X.acp.data[folds!=i,1:j]))
    qda.acp.pred <- predict(qda.acp,newdata=as.data.frame(X.acp.data[folds==i,1:j]))
    qda.acp.perf <- table(y.app[folds==i],qda.acp.pred$class)
    qda.acp.error[j-1] <-qda.acp.error[j-1] + 1 - sum(diag(qda.acp.perf))/numberTest
    
    # Forward
    qda.forward <- qda(y.app[folds!=i]~., data=as.data.frame(X.app[folds!=i,c(reg.fit$vorder[1:j])]))
    qda.forward.pred <- predict(qda.forward,newdata=as.data.frame(X.app[folds==i,c(reg.fit$vorder[1:j])]))
    qda.forward.perf <- table(y.app[folds==i],qda.forward.pred$class)
    qda.forward.error[j-1] <-qda.forward.error[j-1] + 1 - sum(diag(qda.forward.perf))/numberTest
  }
  
  # FDA
  qda.lda <- qda(y.app[folds!=i]~., data=as.data.frame(X.lda.data[folds!=i,]))
  qda.lda.pred <- predict(qda.lda,newdata=as.data.frame(X.lda.data[folds==i,]))
  qda.lda.perf <- table(y.app[folds==i],qda.lda.pred$class)
  qda.lda.error <-qda.lda.error + 1 - sum(diag(qda.lda.perf))/numberTest
  

}

# Diviser le taux d'erreur par le nombre de K 
qda.acp.error <-(qda.acp.error/K)*100
qda.lda.error <-(qda.lda.error/K)*100
qda.forward.error <-(qda.forward.error/K)*100

print(qda.acp.error)
print(qda.lda.error)
print(qda.forward.error)

## === 
## Transformation aussi de nos donnees de test dans le meme referentiel pour toutes 
## ===

# 1) FDA
X.lda.transform.test <- X.test%*%X.lda$scaling

# 2) ACP
X.acp.test.data <- X.test %*% X.acp$rotation

# 3) Forward/Backward Selection
# X.forward.test <- X.test[,which(reg.fit$vorder < ind)]


