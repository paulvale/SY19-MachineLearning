# LOAD PACKAGES 
# =============
load("~/Documents/UTC/A16/SY19/TPs_Desktop/TP/TP7/tp7/data/data_expressions.RData")
library(fda)
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
print("LDA Cumule")
print(X.lda.prop)

X.lda.transform <- X.app%*%X.lda$scaling
pairs(~.,data=X.lda.transform, col=y.app, main="Simple Scatterplot Matrix")

# 2) ACP
X.acp <- prcomp(X.app)
plot(X.acp)
acp.sum <- cumsum(100 * X.acp$sdev^2 / sum(X.acp$sdev^2))
print("ACP Cumule")
plot(acp.sum)

X.acp.transform1 <- X.acp$x[,which(acp.sum<75)]
X.acp.transform2 <- X.acp$x[,which(acp.sum<85)]

# 3) Forward/Backward Selection
# Backward selection requires that the number of samples n is larger than the number of variables p
# (so that the full model can be fit). In contrast, forward stepwise can be used even when n < p,
# and so is the only viable subset method when p is very large.
# ==> Only Forward Selection
reg.fit<-regsubsets(y.app~.,data=as.data.frame(X.app),method="forward",nvmax=50)
# plot(reg.fit,scale="r2")

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

X.acp.transform1 <- X.acp.test[,which(acp.sum<75)]
X.acp.transform2 <- X.acp.test[,which(acp.sum<85)]

# 3) Forward/Backward Selection
X.forward.test <- X.test[,which(reg.fit$vorder < ind)]





