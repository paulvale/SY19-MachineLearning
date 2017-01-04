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
# Logistic Regression
# ===
K <- 5 # Nombre de sections dans notre ensemble d'apprentissage
folds <- sample(1:K,X.app.dim[1] ,replace=TRUE)
logReg.acpF.error <- 0
logReg.acp1.error <- 0
logReg.acp2.error <- 0
logReg.lda.error <- 0
logReg.forward.error <- 0

for(i in 1:K){
  numberTest <- dim(X.acp.transform1[folds==i,])[1]
  logReg.acpF.res<-c(rep(0,numberTest))
  logReg.acp1.res<-c(rep(0,numberTest))
  logReg.acp2.res<-c(rep(0,numberTest))
  logReg.lda.res<-c(rep(0,numberTest))
  logReg.forward.res<-c(rep(0,numberTest))

  logReg.acpF <- glmnet(X.acp.full[folds!=i,] ,y=y.app[folds!=i],family="multinomial")
  logReg.acpF.pred <- predict(logReg.acpF,newx=X.acp.full[folds==i,],type="response",s=logReg.acpF$lambda.min)
    
  logReg.acp1 <- glmnet(X.acp.transform1[folds!=i,] ,y=y.app[folds!=i],family="multinomial")
  logReg.acp1.pred <- predict(logReg.acp1,newx=X.acp.transform1[folds==i,],type="response",s=logReg.acp1$lambda.min)
  
  logReg.acp2 <- glmnet(X.acp.transform2[folds!=i,] ,y=y.app[folds!=i],family="multinomial")
  logReg.acp2.pred <- predict(logReg.acp2,newx=X.acp.transform2[folds==i,],type="response",s=logReg.acp2$lambda.min)
  
  logReg.lda <- glmnet(X.lda.transform[folds!=i,] ,y=y.app[folds!=i],family="multinomial")
  logReg.lda.pred <- predict(logReg.lda,newx=X.lda.transform[folds==i,],type="response",s=logReg.lda$lambda.min)
  
  logReg.forward <- glmnet(X.forward[folds!=i,] ,y=y.app[folds!=i],family="multinomial")
  logReg.forward.pred <- predict(logReg.forward,newx=X.forward[folds==i,],type="response",s=logReg.forward$lambda.min)
  
  for (h in 1:numberTest) {
    logReg.acpF.res[h] <-which.max(logReg.acpF.pred[h,1:6,dim(logReg.acpF.pred)[3]])
    logReg.acp1.res[h] <-which.max(logReg.acp1.pred[h,1:6,dim(logReg.acp1.pred)[3]])
    logReg.acp2.res[h] <-which.max(logReg.acp2.pred[h,1:6,dim(logReg.acp2.pred)[3]])
    logReg.lda.res[h] <-which.max(logReg.lda.pred[h,1:6,dim(logReg.lda.pred)[3]])
    logReg.forward.res[h] <-which.max(logReg.forward.pred[h,1:6,dim(logReg.forward.pred)[3]])
  }
  logReg.acpF.perf <- table(y.app[folds==i],logReg.acpF.res)
  logReg.acpF.error <-logReg.acpF.error + 1 - sum(diag(logReg.acpF.perf))/numberTest
  
  logReg.acp1.perf <- table(y.app[folds==i],logReg.acp1.res)
  logReg.acp1.error <-logReg.acp1.error + 1 - sum(diag(logReg.acp1.perf))/numberTest
  
  logReg.acp2.perf <- table(y.app[folds==i],logReg.acp2.res)
  logReg.acp2.error <-logReg.acp2.error + 1 - sum(diag(logReg.acp2.perf))/numberTest
  
  logReg.lda.perf <- table(y.app[folds==i],logReg.lda.res)
  logReg.lda.error <-logReg.lda.error + 1 - sum(diag(logReg.lda.perf))/numberTest
  
  logReg.forward.perf <- table(y.app[folds==i],logReg.forward.res)
  logReg.forward.error <-logReg.forward.error + 1 - sum(diag(logReg.forward.perf))/numberTest
}

# Diviser le taux d'erreur par le nombre de K 
logReg.acpF.error <-(logReg.acpF.error/K)*100
logReg.acp1.error <-(logReg.acp1.error/K)*100
logReg.acp2.error <-(logReg.acp2.error/K)*100
logReg.lda.error <-(logReg.lda.error/K)*100
logReg.forward.error <-(logReg.forward.error/K)*100

print(logReg.acpF.error)
print(logReg.acp1.error)
print(logReg.acp2.error)
print(logReg.lda.error)
print(logReg.forward.error)

