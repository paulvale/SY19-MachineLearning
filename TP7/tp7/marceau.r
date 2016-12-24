
# =============
rm(list=ls())
load("data/data_expressions.RData")
library(MASS)
library(leaps)
library(class)
library("neuralnet")

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

# ----------------------------
# ----- Neural network -------
# ----------------------------

traininginput <-  as.data.frame(X.acp.transform1)
ideal <- class.ind(y.app)

# --------------------- this is the method for a regression problem -----------------------------------



#Column bind the data into one variable
#trainingdata <- cbind(traininginput,trainingoutput)
#Train the neural network
#Going to have 10 hidden layers
#Threshold is a numeric value specifying the threshold for the partial
#derivatives of the error function as stopping criteria.
#net.sqrt <- neuralnet(trainingdata[,25]~trainingdata[,1]+trainingdata[,2]+trainingdata[,3]+trainingdata[,4]+trainingdata[,5]+trainingdata[,6]+trainingdata[,7]+trainingdata[,8]+trainingdata[,9]+trainingdata[,10]+trainingdata[,11]+trainingdata[,12]+trainingdata[,13]+trainingdata[,14]+trainingdata[,15]+trainingdata[,16]+trainingdata[,17]+trainingdata[,18]+trainingdata[,19]+trainingdata[,20]+trainingdata[,21]+trainingdata[,22]+trainingdata[,23]+trainingdata[,24],trainingdata, hidden=10, threshold=0.01)
#print(net.sqrt)
 
#Plot the neural network
#plot(net.sqrt)
 
#Test the neural network on some training data
#net.results <- compute(net.sqrt, X.acp.test.transform1) #Run them through the neural network
 
#Lets see what properties net.sqrt has
#ls(net.results)
 
#Lets see the results
#print(net.results$net.result)
 

 # --------------------- this is the method for a classification problem -----------------------------------


# ------------ ACP -----------------

# constructing the model with our trainnig dataset
acp1.fit.net <- nnet(X.acp.transform1, ideal, size=10, softmax=TRUE)
# predicting the datas of the testing dataset
acp1.predict.net <- predict(acp1.fit.net, X.acp.test.transform1, type="class")
#evaluating the performance of the classification of the neural network
acp1.perf.net <- table(acp1.predict.net,y.test)
acp1.error.net <- 1-sum(diag(acp1.perf.net))/length(y.test)
print(acp1.error.net)


# constructing the model with our trainnig dataset
acp2.fit.net <- nnet(X.acp.transform2, ideal, size=10, softmax=TRUE)
# predicting the datas of the testing dataset
acp2.predict.net <- predict(acp2.fit.net, X.acp.test.transform2, type="class")
#evaluating the performance of the classification of the neural network
acp2.perf.net <- table(acp2.predict.net,y.test)
acp2.error.net <- 1-sum(diag(acp2.perf.net))/length(y.test)
print(acp2.error.net)


# ------------- FDA -----------------

# constructing the model with our trainnig dataset
lda.fit.net <- nnet(X.lda.transform, ideal, size=10, softmax=TRUE)
# predicting the datas of the testing dataset
lda.predict.net <- predict(lda.fit.net, X.lda.transform.test, type="class")
#evaluating the performance of the classification of the neural network
lda.perf.net <- table(lda.predict.net,y.test)
lda.error.net <- 1-sum(diag(lda.perf.net))/length(y.test)
print(lda.error.net)


# ------------- Forward -----------------

# constructing the model with our trainnig dataset
forward.fit.net <- nnet(X.forward, ideal, size=10, softmax=TRUE)
# predicting the datas of the testing dataset
forward.predict.net <- predict(forward.fit.net, X.forward.test, type="class")
#evaluating the performance of the classification of the neural network
forward.perf.net <- table(forward.predict.net,y.test)
forward.error.net <- 1-sum(diag(forward.perf.net))/length(y.test)
print(forward.error.net)

