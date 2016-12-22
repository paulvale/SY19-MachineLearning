# LOAD PACKAGES 
# =============
rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/data/data_expressions.RData")
library(MASS)
library(leaps)
library(glmnet)
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


K <- 2 # Nombre de sections dans notre ensemble d'apprentissage
numberKnn <- 40
folds <- sample(1:K,X.app.dim[1] ,replace=TRUE)

lda.acp.error <- rep(0,159)
lda.lda.error <- 0
lda.forward.error <- rep(0, X.app.dim[2])

qda.acp.error <- rep(0,5)
qda.lda.error <- 0
qda.forward.error <- rep(0,5)

knn.acp.error <- matrix(0,nrow=numberKnn, ncol=159)
knn.lda.error <- rep(0,numberKnn)
knn.forward.error <- matrix(0,nrow=numberKnn, ncol=X.app.dim[2])

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
  reg.fit<-regsubsets(y.app~.,data=as.data.frame(X.app),method="forward")

  numberTest <- dim(X.lda.data[folds==i,])[1]
  
  # ===
  # ACP
  # ===
  for(j in 2:160){
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
    
    # KNN
    #for(number in 1:numberKnn){
      # Forward
     # knn.forward <- knn(as.data.frame(X.app[folds!=i,c(reg.fit$vorder[1:j])]), as.data.frame(X.app[folds==i,c(reg.fit$vorder[1:j])]), y.app[folds!=i],k=number)
    #  knn.forward.perf <- table(y.app[folds==i], knn.forward)
     # knn.forward.error[number,j-1] <-knn.forward.error[number,j-1]  + 1 - sum(diag(knn.forward.perf))/numberTest
    #}
  #}
  
  # ===
  # FDA
  # ===
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

lda.acp.error <-(lda.acp.error/K)*100
lda.lda.error <-(lda.lda.error/K)*100
lda.forward.error <-(lda.forward.error/K)*100

knn.acp.error <- (knn.acp.error/K)*100
knn.lda.error <- (knn.lda.error/K)*100
knn.forward.error <- (knn.forward.error/K)*100

print("QDA :")
print(min(qda.acp.error))
print(min(qda.lda.error))
print(min(qda.forward.error))

print("LDA :")
print(min(lda.acp.error))
print(min(lda.lda.error))
#print(lda.forward.error)

print("KNN :")
print(min(knn.acp.error))
print(min(knn.lda.error))
#print(knn.forward.error)

