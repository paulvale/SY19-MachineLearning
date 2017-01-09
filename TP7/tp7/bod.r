# LOAD PACKAGES 
# =============
rm(list=ls())
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/data/data_expressions.RData")
library(MASS)
library(leaps)
library(corrplot)
library(randomForest)
library(ROCR)


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
# Nous avons donc deja supprime 540 colonnes qui ne nous sont pas utiles ici sans perte d'informations

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

# Observation :
# Afin de garder un nombre egale des differents cas, nous avons bien separe le jeu de donnee en app et test
# cependant, nous avons de plus fait attention a garder un nombre egale de representation des 6 expressions
# Ainsi, on a donc 29 images a chaque fois de chacun de nos expressions 
# et notre ensemble de test et donc de 7 images poour chaque expression


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
# print("LDA Cumule")
# print(X.lda.prop)

X.lda.transform <- X.app%*%X.lda$scaling

# 2) ACP
X.acp <- prcomp(X.app)
plot(X.acp)
acp.sum <- cumsum(100 * X.acp$sdev^2 / sum(X.acp$sdev^2))
 print("ACP Cumule")
plot(acp.sum, xlab="number of principal components", ylab="Total of variance explained", main="ACP Cumule")

X.acp.full <- X.acp$x 
X.acp.transform1 <- X.acp$x[,which(acp.sum<75)]
X.acp.transform2 <- X.acp$x[,which(acp.sum<85)]

# 3) Forward/Backward Selection
# Backward selection requires that the number of samples n is larger than the number of variables p
# (so that the full model can be fit). In contrast, forward stepwise can be used even when n < p,
# and so is the only viable subset method when p is very large.
# ==> Only Forward Selection
reg.fit<-regsubsets(y.app~.,data=as.data.frame(X.app),method="forward",nvmax=50)

tmp <- which(reg.fit$rss > 50)
ind <- tmp[length(tmp)]

X.forward <- X.app[,which(reg.fit$vorder < ind)]

# Maintenant on va pouvoir se lancer dans les differents modeles
# avec nos differents jeux de donnees 
# Different modeles a essayer :
# - Random Forest
# - SVM
# - NN
# - Classsification naive bayesienne
# - Reg logistique

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


## ===
## VISUALISATION DES DONNEES
## ===

# 1) FDA
pairs(~.,data=X.lda.transform, col=y.app, main="LDA Data")
par(mfrow = c(1, 1))
boxplot(X.lda.transform[,1]~y.app, main="LDA Data", 
        xlab="LD1 value", ylab="faces expression value")

# 2) ACP
pairs(~.,data=X.acp.transform1[,1:5], col=y.app, main="ACP Data")
par(mfrow = c(1, 1))
boxplot(X.acp.transform1[,1]~y.app, main="ACP Data", 
        xlab="PC1 value", ylab="faces expression value")

# 3) Forward Selection
pairs(~.,data=X.forward[,1:5], col=y.app, main="Forward Selection Data")
X.forward.cor <- cor(X.forward)
par(mfrow = c(1, 1))
corrplot(X.forward.cor, type="lower", tl.cex = 0.6)
boxplot(X.forward[,1]~y.app, main="Forward Selection Data", 
        xlab="Forward1 value", ylab="faces expression value")

## ===
## Random Forest
## ===
# calcul des taux d'erreur
vectorTree <- seq(100,1500,100)

# ===
# acpFull
# ===
# err
rF.acpF.tauxErreur.test <- rep(0,length(vectorTree))
rF.acpF.tauxErreur.app <- rep(0,length(vectorTree))

# precision
rF.acpF.precision.test <- rep(0,length(vectorTree))
rF.acpF.precision.app <- rep(0,length(vectorTree))

# rappel
rF.acpF.rappel.test <- rep(0,length(vectorTree))
rF.acpF.rappel.app <- rep(0,length(vectorTree))

# ===
# acp1
# ===
# err
rF.acp1.tauxErreur.test <- rep(0,length(vectorTree))
rF.acp1.tauxErreur.app <- rep(0,length(vectorTree))

# precision
rF.acp1.precision.test <- rep(0,length(vectorTree))
rF.acp1.precision.app <- rep(0,length(vectorTree))

# rappel
rF.acp1.rappel.test <- rep(0,length(vectorTree))
rF.acp1.rappel.app <- rep(0,length(vectorTree))

# ===
# acp2
# ===
# err
rF.acp2.tauxErreur.test <- rep(0,length(vectorTree))
rF.acp2.tauxErreur.app <- rep(0,length(vectorTree))

# precision
rF.acp2.precision.test <- rep(0,length(vectorTree))
rF.acp2.precision.app <- rep(0,length(vectorTree))

# rappel
rF.acp2.rappel.test <- rep(0,length(vectorTree))
rF.acp2.rappel.app <- rep(0,length(vectorTree))


# ===
# forward
# ===
# err
rF.forward.tauxErreur.test <- rep(0,length(vectorTree))
rF.forward.tauxErreur.app <- rep(0,length(vectorTree))

# precision
rF.forward.precision.test <- rep(0,length(vectorTree))
rF.forward.precision.app <- rep(0,length(vectorTree))

# rappel
rF.forward.rappel.test <- rep(0,length(vectorTree))
rF.forward.rappel.app <- rep(0,length(vectorTree))

# ===
# lda
# ===
# err
rF.lda.tauxErreur.test <- rep(0,length(vectorTree))
rF.lda.tauxErreur.app <- rep(0,length(vectorTree))

# preicision
rF.lda.precision.test <- rep(0,length(vectorTree))
rF.lda.precision.app <- rep(0,length(vectorTree))

# rappel
rF.lda.rappel.test <- rep(0,length(vectorTree))
rF.lda.rappel.app <- rep(0,length(vectorTree))

for(i in vectorTree) {
  print(i)
  rF.acpF <- randomForest(factor(y.app)~., data=as.data.frame(X.acp.full), xtest=as.data.frame(X.acp.test.full), ytest=factor(y.test), ntree=i)
  #print("ACP Full RandomForest :")
  #print(rF.acpF$test$confusion)
  
  rF.acp1 <- randomForest(factor(y.app)~., data=as.data.frame(X.acp.transform1), xtest=as.data.frame(X.acp.test.transform1), ytest=factor(y.test), ntree=i)
  #print("ACP1 RandomForest :")
  #print(rF.acp1$test$confusion)
  
  rF.acp2 <- randomForest(factor(y.app)~., data=as.data.frame(X.acp.transform2), xtest=as.data.frame(X.acp.test.transform2), ytest=factor(y.test), ntree=i)
  #print("ACP1 RandomForest :")
  #print(rF.acp2$test$confusion)
  
  rF.lda <- randomForest(factor(y.app)~., data=as.data.frame(X.lda.transform), xtest=as.data.frame(X.lda.transform.test), ytest=factor(y.test), ntree=i)
  #print("LDA RandomForest :")
  #print(rF.lda$test$confusion)
  
  rF.forward <- randomForest(factor(y.app)~., data=as.data.frame(X.forward), xtest=as.data.frame(X.forward.test), ytest=factor(y.test), ntree=i)
  #print("Forward Stepwise RandomForest :")
  #print(rF.forward$test$confusion)
  
  # Calcul des taux d'erreurs
  rF.acpF.tauxErreur.test[i/100] <- 1 - (sum(diag(rF.acpF$test$confusion)/X.test.dim[1]))
  rF.acp1.tauxErreur.test[i/100] <- 1 - (sum(diag(rF.acp1$test$confusion)/X.test.dim[1]))
  rF.acp2.tauxErreur.test[i/100] <- 1 - (sum(diag(rF.acp2$test$confusion)/X.test.dim[1]))
  rF.lda.tauxErreur.test[i/100] <- 1 - (sum(diag(rF.lda$test$confusion)/X.test.dim[1]))
  rF.forward.tauxErreur.test[i/100] <- 1 - (sum(diag(rF.forward$test$confusion)/X.test.dim[1]))
  
  rF.acpF.tauxErreur.app[i/100] <- 1 - (sum(diag(rF.acpF$confusion)/X.app.dim[1]))
  rF.acp1.tauxErreur.app[i/100] <- 1 - (sum(diag(rF.acp1$confusion)/X.app.dim[1]))
  rF.acp2.tauxErreur.app[i/100] <- 1 - (sum(diag(rF.acp2$confusion)/X.app.dim[1]))
  rF.lda.tauxErreur.app[i/100] <- 1 - (sum(diag(rF.lda$confusion)/X.app.dim[1]))
  rF.forward.tauxErreur.app[i/100] <- 1 - (sum(diag(rF.forward$confusion)/X.app.dim[1]))
  
  for( j in 1:6){
    # Calcul de la precision
    rF.acpF.precision.test[i/100] <- rF.acpF.precision.test[i/100] + if(sum(rF.acpF$test$confusion[,j]) == 0) 0 else (rF.acpF$test$confusion[j,j]/sum(rF.acpF$test$confusion[,j]))
    rF.acp1.precision.test[i/100] <- rF.acp1.precision.test[i/100] + if(sum(rF.acp1$test$confusion[,j]) == 0) 0 else (rF.acp1$test$confusion[j,j]/sum(rF.acp1$test$confusion[,j]))
    rF.acp2.precision.test[i/100] <- rF.acp2.precision.test[i/100] + if(sum(rF.acp2$test$confusion[,j]) == 0) 0 else (rF.acp2$test$confusion[j,j]/sum(rF.acp2$test$confusion[,j]))
    rF.lda.precision.test[i/100] <- rF.lda.precision.test[i/100] + if(sum(rF.lda$test$confusion[,j]) == 0) 0 else (rF.lda$test$confusion[j,j]/sum(rF.lda$test$confusion[,j]))
    rF.forward.precision.test[i/100] <- rF.forward.precision.test[i/100] + if(sum(rF.forward$test$confusion[,j]) == 0) 0 else (rF.forward$test$confusion[j,j]/sum(rF.forward$test$confusion[,j]))

    rF.acpF.precision.app[i/100] <- rF.acpF.precision.app[i/100] + if(sum(rF.acpF$confusion[,j]) == 0) 0 else (rF.acpF$confusion[j,j]/sum(rF.acpF$confusion[,j]))
    rF.acp1.precision.app[i/100] <- rF.acp1.precision.app[i/100] + if(sum(rF.acp1$confusion[,j]) == 0) 0 else (rF.acp1$confusion[j,j]/sum(rF.acp1$confusion[,j]))
    rF.acp2.precision.app[i/100] <- rF.acp2.precision.app[i/100] + if(sum(rF.acp2$confusion[,j]) == 0) 0 else (rF.acp2$confusion[j,j]/sum(rF.acp2$confusion[,j]))
    rF.lda.precision.app[i/100] <- rF.lda.precision.app[i/100] + if(sum(rF.lda$confusion[,j]) == 0) 0 else (rF.lda$confusion[j,j]/sum(rF.lda$confusion[,j]))
    rF.forward.precision.app[i/100] <- rF.forward.precision.app[i/100] + if(sum(rF.forward$confusion[,j]) == 0) 0 else (rF.forward$confusion[j,j]/sum(rF.forward$confusion[,j]))
    
    # Calcul du rappel
  }
  rF.acpF.precision.test[i/100] <- (rF.acpF.precision.test[i/100]/6)*100
  rF.acp1.precision.test[i/100] <- (rF.acp1.precision.test[i/100]/6)*100
  rF.acp2.precision.test[i/100] <- (rF.acp2.precision.test[i/100]/6)*100
  rF.lda.precision.test[i/100] <- (rF.lda.precision.test[i/100]/6)*100
  rF.forward.precision.test[i/100] <- (rF.forward.precision.test[i/100]/6)*100
  
  rF.acpF.precision.app[i/100] <- (rF.acpF.precision.app[i/100]/6)*100
  rF.acp1.precision.app[i/100] <- (rF.acp1.precision.app[i/100]/6)*100
  rF.acp2.precision.app[i/100] <- (rF.acp2.precision.app[i/100]/6)*100
  rF.lda.precision.app[i/100] <- (rF.lda.precision.app[i/100]/6)*100
  rF.forward.precision.app[i/100] <- (rF.forward.precision.app[i/100]/6)*100
  
}
# ===
# Graphes
# ===

# Taux d'erreur
# acp full
min <- min(c(rF.acpF.tauxErreur.app, rF.acpF.tauxErreur.test))
max <- max(c(rF.acpF.tauxErreur.app, rF.acpF.tauxErreur.test))
plot(vectorTree,rF.acpF.tauxErreur.app,type="l",col="red", ylim=c(min,max), ylab="")
par(new=TRUE)
plot(vectorTree,rF.acpF.tauxErreur.test,col="green", type="l", ylim=c(min,max), ylab="taux d'erreur", main="Taux erreur ACP Full")


# acp1
min <- min(c(rF.acp1.tauxErreur.app, rF.acp1.tauxErreur.test))
max <- max(c(rF.acp1.tauxErreur.app, rF.acp1.tauxErreur.test))
plot(vectorTree,rF.acp1.tauxErreur.app,type="l",col="red", ylim=c(min,max), ylab="")
par(new=TRUE)
plot(vectorTree,rF.acp1.tauxErreur.test,col="green", type="l", ylim=c(min,max), ylab="taux d'erreur", main="Taux erreur ACP 1")

# acp2
min <- min(c(rF.acp2.tauxErreur.app, rF.acp2.tauxErreur.test))
max <- max(c(rF.acp2.tauxErreur.app, rF.acp2.tauxErreur.test))
plot(vectorTree,rF.acp2.tauxErreur.app,type="l",col="red", ylim=c(min,max), ylab="")
par(new=TRUE)
plot(vectorTree,rF.acp2.tauxErreur.test,col="green", type="l", ylim=c(min,max), ylab="taux d'erreur", main="Taux erreur ACP 2")

# lda
min <- min(c(rF.lda.tauxErreur.app, rF.lda.tauxErreur.test))
max <- max(c(rF.lda.tauxErreur.app, rF.lda.tauxErreur.test))
plot(vectorTree,rF.lda.tauxErreur.app,type="l",col="red", ylim=c(min,max), ylab="")
par(new=TRUE)
plot(vectorTree,rF.lda.tauxErreur.test,col="green", type="l", ylim=c(min,max), ylab="taux d'erreur", main="Taux erreur LDA")

# forward
min <- min(c(rF.forward.tauxErreur.app, rF.forward.tauxErreur.test))
max <- max(c(rF.forward.tauxErreur.app, rF.forward.tauxErreur.test))
plot(vectorTree,rF.forward.tauxErreur.app,type="l",col="red", ylim=c(min,max), ylab="")
par(new=TRUE)
plot(vectorTree,rF.forward.tauxErreur.test,col="green", type="l", ylim=c(min,max), ylab="taux d'erreur", main="Taux erreur Forward")


# Precision
# acp full
min <- min(c(rF.acpF.precision.app, rF.acpF.precision.test))
max <- max(c(rF.acpF.precision.app, rF.acpF.precision.test))
plot(vectorTree,rF.acpF.precision.app,type="l",col="red", ylim=c(min,max), ylab="")
par(new=TRUE)
plot(vectorTree,rF.acpF.precision.test,col="green", type="l", ylim=c(min,max), ylab="%", main="Precision ACP Full")


# acp1
min <- min(c(rF.acp1.precision.app, rF.acp1.precision.test))
max <- max(c(rF.acp1.precision.app, rF.acp1.precision.test))
plot(vectorTree,rF.acp1.precision.app,type="l",col="red", ylim=c(min,max), ylab="")
par(new=TRUE)
plot(vectorTree,rF.acp1.precision.test,col="green", type="l", ylim=c(min,max), ylab="%", main="Precision ACP 1")

# acp2
min <- min(c(rF.acp2.precision.app, rF.acp2.precision.test))
max <- max(c(rF.acp2.precision.app, rF.acp2.precision.test))
plot(vectorTree,rF.acp2.precision.app,type="l",col="red", ylim=c(min,max), ylab="")
par(new=TRUE)
plot(vectorTree,rF.acp2.precision.test,col="green", type="l", ylim=c(min,max), ylab="%", main="Precision ACP 2")

# lda
min <- min(c(rF.lda.precision.app, rF.lda.precision.test))
max <- max(c(rF.lda.precision.app, rF.lda.precision.test))
plot(vectorTree,rF.lda.precision.app,type="l",col="red", ylim=c(min,max), ylab="")
par(new=TRUE)
plot(vectorTree,rF.lda.precision.test,col="green", type="l", ylim=c(min,max), ylab="%", main="Precision LDA")

# forward
min <- min(c(rF.forward.precision.app, rF.forward.precision.test))
max <- max(c(rF.forward.precision.app, rF.forward.precision.test))
plot(vectorTree,rF.forward.precision.app,type="l",col="red", ylim=c(min,max), ylab="")
par(new=TRUE)
plot(vectorTree,rF.forward.precision.test,col="green", type="l", ylim=c(min,max), ylab="%", main="Precision Forward")

# Observations faites sur les graphes :
# on se rend comtpe ici que forward est le seule algos qui nous donnent une erreur d'apprentissage 
# superieur a notre erreur de test

# Cependant dans tous les cas il est tres eleve ( min 65% taux d'erreur)
# => LDA
# ntree = 200 
# taux erreur app : 5%
#             test : 28%
# est ce que on ne serait pas en overfitting ?

# => ACP1
# ntree = 200 
# taux erreur app : 28%
#             test : 50%

# => ACP2
# ntree = 100 
# taux erreur app : 31%
#             test : 45%

# ===> Notre meilleur modele c'est le LDA pour le moment avec les random forests
