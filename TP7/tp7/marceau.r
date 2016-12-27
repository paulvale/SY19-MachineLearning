
# =============
rm(list=ls())
load("data/data_expressions.RData")
library(MASS)
library(leaps)
library(class)
library("neuralnet")
library(nnet)

getFormulas <- function(col, order, label) {
  result <- vector(mode="character", length=length(order))
  for(i in 1:length(order)){
    if(i == 1){
      result[i] <- paste(c(as.character(label)," ~ ",col[order[i]]),collapse = '')
    } else {
      result[i] <- paste(c(result[i-1], col[order[i]]), collapse = ' + ')
    }
  }
  return(result)
}

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




# pour la fonciton neuralnet()
# cette ffonction est seulement utile lorsuq'il y a deux classes.
data.train = X.lda.transform
data.test = X.lda.transform.test
ordre = c(1:dim(data.train)[2])


misClassificationError1 = rep(0,10)
misClassificationError2 = rep(0,10)
misClassificationError3 = rep(0,10)
misClassificationError4 = rep(0,10)
misClassificationError5 = rep(0,10)
misClassificationError6 = rep(0,10)
neuralnet.error = rep(0,10)


for(j in 1:10){
	for(k in 1:6){
		response <- rep(0,length(y.app))
		for(i in 1:length(y.app)){
			if(y.app[i]==k){
				response[i]=1
			}
		}
		newDataSet <- data.frame(data.train,response)
		formule = getFormulas(colnames(newDataSet), ordre,"response")
		if(k==1)
		{
			print("1/6")
			neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,err.fct="ce",linear.output=FALSE)
			nn1 <- ifelse(neuralnet1$net.result[[1]]>0.5,1,0)
			misClassificationError1[j] <- mean(response!=nn1)
			print(misClassificationError1[j])
		}
		else if( k==2)
		{
			print("2/6")
			neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,err.fct="ce",linear.output=FALSE)
			nn2 <- ifelse(neuralnet2$net.result[[1]]>0.5,1,0)
			misClassificationError2[j] <- mean(response!=nn2)
			print(misClassificationError2[j])
		}
		else if(k==3){
			print("3/6")
			neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,err.fct="ce",linear.output=FALSE)
			nn3 <- ifelse(neuralnet3$net.result[[1]]>0.5,1,0)
			misClassificationError3[j] <- mean(response!=nn3)
			print(misClassificationError3[j])
		}
		else if(k==4){
			print("4/6")
			neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,err.fct="ce",linear.output=FALSE)
			nn4 <- ifelse(neuralnet4$net.result[[1]]>0.5,1,0)
			misClassificationError4[j] <- mean(response!=nn4)
			print(misClassificationError4[j])
		}
		else if(k==5){
			print("5/6")
			neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,err.fct="ce",linear.output=FALSE)
			nn5 <- ifelse(neuralnet5$net.result[[1]]>0.5,1,0)
			misClassificationError5[j] <- mean(response!=nn5)
			print(misClassificationError5[j])
		}
		else if(k==6){
			print("6/6")
			neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,err.fct="ce",linear.output=FALSE)
			nn6 <- ifelse(neuralnet6$net.result[[1]]>0.5,1,0)
			misClassificationError6[j] <- mean(response!=nn6)
			print(misClassificationError6[j])
		}
		
	}

	# predictions 

	c1<-compute(neuralnet1,as.matrix(data.test))
	c2<-compute(neuralnet2,as.matrix(data.test))
	c3<-compute(neuralnet3,as.matrix(data.test))
	c4<-compute(neuralnet4,as.matrix(data.test))
	c5<-compute(neuralnet5,as.matrix(data.test))
	c6<-compute(neuralnet6,as.matrix(data.test))

	neuralnet.result <- rep(0,dim(data.test)[1])

	for(i in 1:dim(data.test)[1]){
		index <- which.max(c(c1$net.result[i],c2$net.result[i],c3$net.result[i],c4$net.result[i],c5$net.result[i],c6$net.result[i]))
		neuralnet.result[i] <- index
	}
	neuralnet.perf <- table(neuralnet.result,y.test)
	neuralnet.error[j] <- 1-sum(diag(neuralnet.perf))/length(y.test)
	print(neuralnet.error[j])
}
print(mean(neuralnet.error))


#confidence interval 

#ci <- confidence.interval(neuralnet1,alpha=0.05)



# neuralnet <- neuralnet(formule[5],newDataSet,hidden=c(2,3),linear.output=FALSE)
# pour un probleme de regression il faut le mettre le critere linear.output a TRUE
# pour un probleme de clasification il faut mettre le critere linear.output a FALSE
# il faut utiliser la crossvalidation pour trouver le best hiddent layer possible
# comme on utilise des vagues aleatoires il se peut que de temps en temps on ne puisse pas converger alors que des fois si


# essayons maintenant avec l'algorithme de backproagation
print("let s go for backpropagation")



for(j in 1:10){
	for(k in 1:6){
		response <- rep(0,length(y.app))
		for(i in 1:length(y.app)){
			if(y.app[i]==k){
				response[i]=1
			}
		}
		newDataSet <- data.frame(data.train,response)
		formule = getFormulas(colnames(newDataSet), ordre,"response")
		if(k==1)
		{
			print("1/6")
			neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
			nn1 <- ifelse(neuralnet1$net.result[[1]]>0.5,1,0)
			misClassificationError1[j] <- mean(response!=nn1)
		}
		else if( k==2)
		{
			print("2/6")
			neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
			nn2 <- ifelse(neuralnet2$net.result[[1]]>0.5,1,0)
			misClassificationError2[j] <- mean(response!=nn2)
		}
		else if(k==3){
			print("3/6")
			neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
			nn3 <- ifelse(neuralnet3$net.result[[1]]>0.5,1,0)
			misClassificationError3[j] <- mean(response!=nn3)
		}
		else if(k==4){
			print("4/6")
			neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
			nn4 <- ifelse(neuralnet4$net.result[[1]]>0.5,1,0)
			misClassificationError4[j] <- mean(response!=nn4)
		}
		else if(k==5){
			print("5/6")
			neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
			nn5 <- ifelse(neuralnet5$net.result[[1]]>0.5,1,0)
			misClassificationError5[j] <- mean(response!=nn5)
		}
		else if(k==6){
			print("6/6")
			neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=6,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
			nn6 <- ifelse(neuralnet6$net.result[[1]]>0.5,1,0)
			misClassificationError6[j] <- mean(response!=nn6)
		}
		
	}
	# predictions 

	c1<-compute(neuralnet1,as.matrix(data.test))
	c2<-compute(neuralnet2,as.matrix(data.test))
	c3<-compute(neuralnet3,as.matrix(data.test))
	c4<-compute(neuralnet4,as.matrix(data.test))
	c5<-compute(neuralnet5,as.matrix(data.test))
	c6<-compute(neuralnet6,as.matrix(data.test))

	neuralnet.result <- rep(0,dim(data.test)[1])

	for(i in 1:dim(data.test)[1]){
		index <- which.max(c(c1$net.result[i],c2$net.result[i],c3$net.result[i],c4$net.result[i],c5$net.result[i],c6$net.result[i]))
		neuralnet.result[i] <- index
	}
	neuralnet.perf <- table(neuralnet.result,y.test)
	neuralnet.error[j] <- 1-sum(diag(neuralnet.perf))/length(y.test)
	print(neuralnet.error[j])
}
print(mean(neuralnet.error))




