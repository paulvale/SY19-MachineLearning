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


K <- 2 # Nombre de sections dans notre ensemble d'apprentissage
numberKnn <- 40
vectorTree <- seq(100,1500,100)
folds <- sample(1:K,X.app.dim[1] ,replace=TRUE)


for(i in 1:K){
  print(i)
  # ====
  # Reduction de la dimensionnalite des variables
  # ===
  # 1) FDA
  X.lda <- lda(y.app~., data=as.data.frame(X.app))
  X.lda.data <- X.app%*%X.lda$scaling
  
  # 2) ACP
  X.acp <- prcomp(X.app)
  X.acp.data <- X.acp$x
  
  # 3) Forward/Backward Selection
  reg.fit<-regsubsets(factor(y.app)~.,data=as.data.frame(X.acp.data),method="forward")
  plot(reg.fit)
  
  numberTest <- dim(X.lda.data[folds==i,])[1]
  
  X.forward.data <- X.acp.data[,which(reg.fit$vorder == 1)]
  X.forward.data <- as.data.frame(X.forward.data)
  names(X.forward.data)[1] <- colnames(X.acp.data)[which(reg.fit$vorder == 1)]
  
  # ===
  # Forward
  # ===
  for(j in 2:160){
    X.forward.data[[colnames(X.acp.data)[which(reg.fit$vorder == j)]]] <- X.acp.data[,which(reg.fit$vorder == j)]
  }
  
  # ===
  # Forward
  # ===
  #for(j in 2:X.app.dim[2]){
  # LDA
  # print(j)
  #lda.forward <- lda(y.app[folds!=i]~., data=as.data.frame(X.app[folds!=i,c(reg.fit$vorder[1:j])]))
  #lda.forward.pred <- predict(lda.forward,newdata=as.data.frame(X.app[folds==i,c(reg.fit$vorder[1:j])]))
  #lda.forward.perf <- table(y.app[folds==i],lda.forward.pred$class)
  #lda.forward.error[j-1] <-lda.forward.error[j-1] + 1 - sum(diag(lda.forward.perf))/numberTest
  
  # ===
  # FDA
  # ===
  # LDA
  #lda.lda <- lda(y.app[folds!=i]~., data=as.data.frame(X.lda.data[folds!=i,]))
  #lda.lda.pred <- predict(lda.lda,newdata=as.data.frame(X.lda.data[folds==i,]))
  #lda.lda.perf <- table(y.app[folds==i],lda.lda.pred$class)
  #lda.lda.error <-lda.lda.error + 1 - sum(diag(lda.lda.perf))/numberTest
  
}





