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
lda.acpF.error <- 0
lda.acp1.error <- 0
lda.acp2.error <- 0
lda.lda.error <- 0
lda.forward.error <- 0

for(i in 1:K){
  numberTest <- dim(X.acp.transform1[folds==i,])[1]

  # ACP Full
  lda.acpF <- lda(y.app[folds!=i]~., data=as.data.frame(X.acp.full[folds!=i,]))
  lda.acpF.pred <- predict(lda.acpF,newdata=as.data.frame(X.acp.full[folds==i,]))
  lda.acpF.perf <- table(y.app[folds==i],lda.acpF.pred$class)
  lda.acpF.error <-lda.acpF.error + 1 - sum(diag(lda.acpF.perf))/numberTest
  
  # ACP 1
  lda.acp1 <- lda(y.app[folds!=i]~., data=as.data.frame(X.acp.transform1[folds!=i,]))
  lda.acp1.pred <- predict(lda.acp1,newdata=as.data.frame(X.acp.transform1[folds==i,]))
  lda.acp1.perf <- table(y.app[folds==i],lda.acp1.pred$class)
  lda.acp1.error <-lda.acp1.error + 1 - sum(diag(lda.acp1.perf))/numberTest

  # ACP 2
  lda.acp2 <- lda(y.app[folds!=i]~., data=as.data.frame(X.acp.transform2[folds!=i,]))
  lda.acp2.pred <- predict(lda.acp2,newdata=as.data.frame(X.acp.transform2[folds==i,]))
  lda.acp2.perf <- table(y.app[folds==i],lda.acp2.pred$class)
  lda.acp2.error <-lda.acp2.error + 1 - sum(diag(lda.acp2.perf))/numberTest

  # FDA
  lda.lda <- lda(y.app[folds!=i]~., data=as.data.frame(X.lda.transform[folds!=i,]))
  lda.lda.pred <- predict(lda.lda,newdata=as.data.frame(X.lda.transform[folds==i,]))
  lda.lda.perf <- table(y.app[folds==i],lda.lda.pred$class)
  lda.lda.error <-lda.lda.error + 1 - sum(diag(lda.lda.perf))/numberTest

  # Forward
  lda.forward <- lda(y.app[folds!=i]~., data=as.data.frame(X.forward[folds!=i,]))
  lda.forward.pred <- predict(lda.forward,newdata=as.data.frame(X.forward[folds==i,]))
  lda.forward.perf <- table(y.app[folds==i],lda.forward.pred$class)
  lda.forward.error <-lda.forward.error + 1 - sum(diag(lda.forward.perf))/numberTest
}

# Diviser le taux d'erreur par le nombre de K 
lda.acpF.error <-(lda.acpF.error/K)*100
lda.acp1.error <-(lda.acp1.error/K)*100
lda.acp2.error <-(lda.acp2.error/K)*100
lda.lda.error <-(lda.lda.error/K)*100
lda.forward.error <-(lda.forward.error/K)*100

print(lda.acpF.error)
print(lda.acp1.error)
print(lda.acp2.error)
print(lda.lda.error)
print(lda.forward.error)

#Obersavtion :
# Pour le lda Full je n'ai pas pu tout mettre 
# j'obetnais une erreur me disant que les dernieres colonnes etaient constantes
# donc ca ne fonctionnait pas
# j'ai pris jusqu'a 166 du coup qui correspond ici au max que l'on peut avoir 

