
# =============
rm(list=ls())
load("data/data_expressions.RData")
library(MASS)
library(leaps)
library(class)
library("neuralnet")
library(nnet)
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

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





# pour un probleme de regression il faut le mettre le critere linear.output a TRUE
# pour un probleme de clasification il faut mettre le critere linear.output a FALSE
# il faut utiliser la crossvalidation pour trouver le best hiddent layer possible
# comme on utilise des vagues aleatoires il se peut que de temps en temps on ne puisse pas converger alors que des fois si


# essayons maintenant avec l'algorithme de backproagation -- 31,19% contre 34,76 % pour l'autre methode,  je vais donc garder cette méthode, meme si l'autre est plus performante en terme de temps


# ce qu'il faut faire :
# 	- une validation croisée pour trouver le nombre de neuronnes cachés a garder
# 	- implementer la stepwise forward selection sur l'acp dans notre validation
# 	- une validation croisée pour savoir si la méthoed la plus performante est backprog ou non
# 	- fait ça pour chaque méthode de réduciton de variable, en sachant que pour l'acp et le stepwise il faudra faire des boucles pour tester selon les variabls


K <- 5 # Nombre de sections dans notre ensemble d'apprentissage
folds <- sample(1:K,X.app.dim[1] ,replace=TRUE)
acp1.error.net <- rep(0,13)
acp2.error.net <- rep(0,13)
fda.error.net <- rep(0,13)
forward.error.net.tot <- rep(0,13)
forward.error.net <- matrix(nrow=13, ncol=59,0)
fda.error.neuralnetNoBP <- rep(0,13)
fda.error.neuralnetBP <- rep(0,13)
fda.error.neuralnetNoBP2 <- rep(0,7)
fda.error.neuralnetBP2 <- rep(0,7)
acp1.error.neuralnet <- rep(0,13)

c1<-0
c2<-0
c3<-0
c4<-0
c5<-0
c6<-0

for (neurone in 3:15)
{
	cat("neurone : ", neurone," sur 15\n")
	for(i in 1:K){

		  print("===")
		  print(i)
		  print("===")
		  # ====
		  # Reduction de la dimensionnalite des variables
		  # ===
		  # 1) FDA
		  X.lda <- lda(y.app[folds!=i]~., data=as.data.frame(X.app[folds!=i,]))
		  X.lda.data <- X.app[folds!=i,]%*%X.lda$scaling
		  X.lda.testfold <- X.app[folds==i,]%*%X.lda$scaling

		  numberTest <- dim(X.lda.testfold)[1]
		  ideal <- class.ind(y.app[folds!=i])


		# ---------------------------------------------------
		# ----- Neural network  avec NNET  et softmax -------
		# ---------------------------------------------------

		# ------ ACP 1 ------
		acp1.fit.net <- nnet(X.acp.transform1[folds!=i,], ideal, size=neurone, softmax=TRUE)
		acp1.predict.net <- predict(acp1.fit.net, X.acp.transform1[folds==i,], type="class")
		acp1.perf.net <- table(acp1.predict.net,y.app[folds==i])
		acp1.error.net[neurone-2] <-acp1.error.net[neurone-2] + 1-sum(diag(acp1.perf.net))/numberTest

		# ------ ACP 2 ------
		acp2.fit.net <- nnet(X.acp.transform2[folds!=i,], ideal, size=neurone, softmax=TRUE)
		acp2.predict.net <- predict(acp2.fit.net, X.acp.transform2[folds==i,], type="class")
		acp2.perf.net <- table(acp2.predict.net,y.app[folds==i])
		acp2.error.net[neurone-2] <-acp2.error.net[neurone-2] + 1-sum(diag(acp2.perf.net))/numberTest

		# ------ full ne marche pas -------

		# ------ FDA ------ 
		fda.fit.net <- nnet(X.lda.data, ideal, size=neurone, softmax=TRUE)
		fda.predict.net <- predict(fda.fit.net, X.lda.testfold, type="class")
		fda.perf.net <- table(fda.predict.net,y.app[folds==i])
		fda.error.net[neurone-2] <-fda.error.net[neurone-2] + 1-sum(diag(fda.perf.net))/numberTest

		# ------ FORWARD ------
		for(j in 2: 60){
			X.acp.forward <- X.acp$x[,1:j]
			forward.fit.net <- nnet(X.acp.forward[folds!=i,], ideal, size=neurone, softmax=TRUE)
			forward.predict.net <- predict(forward.fit.net, X.acp.forward[folds==i,], type="class")
			forward.perf.net <- table(forward.predict.net,y.app[folds==i])
			forward.error.net[neurone-2,j-1] <-1-sum(diag(forward.perf.net))/numberTest
		}
		plot(forward.error.net,type="l")
		forward.error.net.tot[neurone-2] <-forward.error.net.tot[neurone-2] + min(forward.error.net[neurone-2,0:59])

		# ----------------------------------------------------
		# ----- Neural network  avec NNET  et sigmoide -------
		# ----------------------------------------------------

		# --------------------------------------------
		# ----- Neural network  avec NEURALNET -------
		# --------------------------------------------

		# ---------- FDA -----------------------
		print("fda avec backprop")
		data.train = X.lda.data
		data.test = X.lda.testfold
		y.train = y.app[folds!=i]
		y.testfold = y.app[folds==i]
		ordre = c(1:dim(data.train)[2])
		neuralnet1<-0
		neuralnet2<-0
		neuralnet3<-0
		neuralnet4<-0
		neuralnet5<-0
		neuralnet6<-0


			for(k in 1:6){
				response <- rep(0,length(y.train))
				for(j in 1:length(y.train)){
					if(y.train[j]==k){
						response[j]=1
					}
				}
				newDataSet <- data.frame(data.train,response)
				formule = getFormulas(colnames(newDataSet), ordre,"response")
				if(k==1)
				{
					neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn1 <- ifelse(neuralnet1$net.result[[1]]>0.5,1,0)

				}
				else if( k==2)
				{
					neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn2 <- ifelse(neuralnet2$net.result[[1]]>0.5,1,0)

				}
				else if(k==3){
					neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn3 <- ifelse(neuralnet3$net.result[[1]]>0.5,1,0)

				}
				else if(k==4){
					neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn4 <- ifelse(neuralnet4$net.result[[1]]>0.5,1,0)

				}
				else if(k==5){
					neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn5 <- ifelse(neuralnet5$net.result[[1]]>0.5,1,0)

				}
				else if(k==6){
					neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn6 <- ifelse(neuralnet6$net.result[[1]]>0.5,1,0)

				}
				
			}

			# predictions 

			if(!is.null(neuralnet1$net.result)){
				c1<-compute(neuralnet1,as.matrix(data.test))
			} else {c1$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet2$net.result)){
				c2<-compute(neuralnet2,as.matrix(data.test))
			} else {c2$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet3$net.result)){
				c3<-compute(neuralnet3,as.matrix(data.test))
			} else {c3$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet4$net.result)){
				c4<-compute(neuralnet4,as.matrix(data.test))
			} else {c4$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet5$net.result)){
				c5<-compute(neuralnet5,as.matrix(data.test))
			} else {c5$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet6$net.result)){
				c6<-compute(neuralnet6,as.matrix(data.test))
			} else {c6$net.result<-rep(0,dim(data.test)[1])}

			neuralnet.result <- rep(0,dim(data.test)[1])

			for(j in 1:dim(data.test)[1]){
				index <- which.max(c(c1$net.result[j],c2$net.result[j],c3$net.result[j],c4$net.result[j],c5$net.result[j],c6$net.result[j]))
				neuralnet.result[j] <- index
			}
			neuralnet.perf <- table(neuralnet.result,y.testfold)
			fda.error.neuralnetBP[neurone-2] <- fda.error.neuralnetBP[neurone-2]+  1-sum(diag(neuralnet.perf))/length(y.testfold)



		# ------------- FDA sans backprop --------------

		print("FDA sans backprog")
		neuralnet1<-0
		neuralnet2<-0
		neuralnet3<-0
		neuralnet4<-0
		neuralnet5<-0
		neuralnet6<-0
			for(k in 1:6){
				response <- rep(0,length(y.train))
				for(j in 1:length(y.train)){
					if(y.train[j]==k){
						response[j]=1
					}
				}
				newDataSet <- data.frame(data.train,response)
				formule = getFormulas(colnames(newDataSet), ordre,"response")
				if(k==1)
				{
					neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone, err.fct="ce",linear.output=FALSE)
					nn1 <- ifelse(neuralnet1$net.result[[1]]>0.5,1,0)

				}
				else if( k==2)
				{
					neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,err.fct="ce",linear.output=FALSE)
					nn2 <- ifelse(neuralnet2$net.result[[1]]>0.5,1,0)

				}
				else if(k==3){
					neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,err.fct="ce",linear.output=FALSE)
					nn3 <- ifelse(neuralnet3$net.result[[1]]>0.5,1,0)

				}
				else if(k==4){
					neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,err.fct="ce",linear.output=FALSE)
					nn4 <- ifelse(neuralnet4$net.result[[1]]>0.5,1,0)

				}
				else if(k==5){
					neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,err.fct="ce",linear.output=FALSE)
					nn5 <- ifelse(neuralnet5$net.result[[1]]>0.5,1,0)

				}
				else if(k==6){
					neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=neurone,err.fct="ce",linear.output=FALSE)
					nn6 <- ifelse(neuralnet6$net.result[[1]]>0.5,1,0)

				}
				
			}

			# predictions 

			if(!is.null(neuralnet1$net.result)){
				c1<-compute(neuralnet1,as.matrix(data.test))
			} else {c1$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet2$net.result)){
				c2<-compute(neuralnet2,as.matrix(data.test))
			} else {c2$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet3$net.result)){
				c3<-compute(neuralnet3,as.matrix(data.test))
			} else {c3$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet4$net.result)){
				c4<-compute(neuralnet4,as.matrix(data.test))
			} else {c4$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet5$net.result)){
				c5<-compute(neuralnet5,as.matrix(data.test))
			} else {c5$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet6$net.result)){
				c6<-compute(neuralnet6,as.matrix(data.test))
			} else {c6$net.result<-rep(0,dim(data.test)[1])}

			neuralnet.result <- rep(0,dim(data.test)[1])

			for(j in 1:dim(data.test)[1]){
				index <- which.max(c(c1$net.result[j],c2$net.result[j],c3$net.result[j],c4$net.result[j],c5$net.result[j],c6$net.result[j]))
				neuralnet.result[j] <- index
			}
			neuralnet.perf <- table(neuralnet.result,y.testfold)
			fda.error.neuralnetNoBP[neurone-2] <-fda.error.neuralnetNoBP[neurone-2]+ 1-sum(diag(neuralnet.perf))/length(y.testfold)


		# ------------- ACP1 --------------
		print("acp1")
		neuralnet1<-0
		neuralnet2<-0
		neuralnet3<-0
		neuralnet4<-0
		neuralnet5<-0
		neuralnet6<-0
		data.train = X.acp.transform1[folds!=i,]
		data.test = X.acp.transform1[folds==i,]
		y.train = y.app[folds!=i]
		y.testfold = y.app[folds==i]
		ordre = c(1:dim(data.train)[2])


			for(k in 1:6){
				response <- rep(0,length(y.train))
				for(j in 1:length(y.train)){
					if(y.train[j]==k){
						response[j]=1
					}
				}
				newDataSet <- data.frame(data.train,response)
				formule = getFormulas(colnames(newDataSet), ordre,"response")
				if(k==1)
				{
					neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
					nn1 <- ifelse(neuralnet1$net.result[[1]]>0.5,1,0)

				}
				else if( k==2)
				{
					neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
					nn2 <- ifelse(neuralnet2$net.result[[1]]>0.5,1,0)

				}
				else if(k==3){
					neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
					nn3 <- ifelse(neuralnet3$net.result[[1]]>0.5,1,0)

				}
				else if(k==4){
					neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
					nn4 <- ifelse(neuralnet4$net.result[[1]]>0.5,1,0)

				}
				else if(k==5){
					neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
					nn5 <- ifelse(neuralnet5$net.result[[1]]>0.5,1,0)

				}
				else if(k==6){
					neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=3,err.fct="ce",linear.output=FALSE)
					nn6 <- ifelse(neuralnet6$net.result[[1]]>0.5,1,0)

				}
				
			}

			# predictions 

			if(!is.null(neuralnet1$net.result)){
				c1<-compute(neuralnet1,as.matrix(data.test))
			} else {c1$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet2$net.result)){
				c2<-compute(neuralnet2,as.matrix(data.test))
			} else {c2$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet3$net.result)){
				c3<-compute(neuralnet3,as.matrix(data.test))
			} else {c3$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet4$net.result)){
				c4<-compute(neuralnet4,as.matrix(data.test))
			} else {c4$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet5$net.result)){
				c5<-compute(neuralnet5,as.matrix(data.test))
			} else {c5$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet6$net.result)){
				c6<-compute(neuralnet6,as.matrix(data.test))
			} else {c6$net.result<-rep(0,dim(data.test)[1])}

			neuralnet.result <- rep(0,dim(data.test)[1])

			for(j in 1:dim(data.test)[1]){
				index <- which.max(c(c1$net.result[j],c2$net.result[j],c3$net.result[j],c4$net.result[j],c5$net.result[j],c6$net.result[j]))
				neuralnet.result[j] <- index
			}
			neuralnet.perf <- table(neuralnet.result,y.testfold)
			acp1.error.neuralnet[neurone-2] <-acp1.error.neuralnet[neurone-2]+ 1-sum(diag(neuralnet.perf))/length(y.testfold)

	}
}

	# Diviser le taux d'erreur par le nombre de K 
	acp1.error.net <-(acp1.error.net/K)*100
	acp2.error.net <-(acp2.error.net/K)*100
	fda.error.net <-(fda.error.net/K)*100
	forward.error.net.tot <-(forward.error.net.tot/K)*100
	fda.error.neuralnetNoBP <- (fda.error.neuralnetNoBP/K)*100
	fda.error.neuralnetBP <- (fda.error.neuralnetBP/K)*100
	acp1.error.neuralnet <- (acp1.error.neuralnet/K)*100


	neuroneNoBP1<- which.min(fda.error.neuralnetNoBP[1:11])+2
	neuroneBP1<-which.min(fda.error.neuralnetBP[1:11])+2

	cat("[NEURALNET] Neurone optimal BP : ",neuroneBP1,", avec une erreur de : ",fda.error.neuralnetBP[neuroneBP1-2])
	cat("[NEURALNET] Neurone optimal No BP : ",neuroneNoBP1,", avec une erreur de : ",fda.error.neuralnetNoBP[neuroneNoBP1-2])
	cat("[NEURALNET] Neurone optimal ACP : ",which.min(acp1.error.neuralnet)+2,", avec une erreur de : ",acp1.error.neuralnet[which.min(acp1.error.neuralnet)])
	cat("[NNET] Neurone optimal pour ACP1 : ",which.min(acp1.error.net)+2,", avec une erreur de : ",acp1.error.net[which.min(acp1.error.net)])
	cat("[NNET] Neurone optimal pour ACP2 : ",which.min(acp2.error.net)+2,", avec une erreur de : ",acp2.error.net[which.min(acp2.error.net)])
	cat("[NNET] Neurone optimal pour Forward : ",which.min(forward.error.net.tot)+2,", avec une erreur de : ",forward.error.net.tot[which.min(forward.error.net.tot)])
	cat("[NNET] Neurone optimal pour FDA : ",which.min(fda.error.net)+2,", avec une erreur de : ",fda.error.net[which.min(fda.error.net)])

# resultats :

# neuralnet :
#- LDA + BP : 26.68472704 avec 13 neurones
#- LDA  : 23.78331472 avec 11 neurones
#- ACP : 44.52610135 avec 9 neurones

# nnet :
#- ACP1 : 42.71477542 avec 10 neurones
#- ACP2 : 41.82108088 avec 13 neurones
#- forward : 27.80793716 avec 13 neurones
#- LDA : 30.69334587 avec 10 neurones

min=10
max=60
plot(c(3:14),acp1.error.net[1:12],type="l", ylim=c(min,max), ylab="", xlab="",col='blue')
par(new=TRUE)
plot(c(3:14),fda.error.net[1:12],type="l", ylim=c(min,max), ylab="", xlab="",col='green')
par(new=TRUE)
plot(c(3:14),forward.error.net.tot[1:12],type="l",  ylim=c(min,max), ylab="", xlab="",col='black')

min=10
max=60
plot(c(3:14),acp1.error.neuralnet[1:12],type="l", ylim=c(min,max), ylab="", xlab="",col='blue')
par(new=TRUE)
plot(c(3:14),fda.error.neuralnetBP[1:12],type="l",  ylim=c(min,max), ylab="", xlab="",col='red')
par(new=TRUE)
plot(c(3:14),fda.error.neuralnetNoBP[1:12],type="l", ylim=c(min,max), ylab="", xlab="",col='green')

for(neurone2 in 3:9)
{
	cat("neurone : ", neurone2," sur 15\n")
	for(i in 1:K){

		  print("===")
		  print(i)
		  print("===")

		  X.lda <- lda(y.app[folds!=i]~., data=as.data.frame(X.app[folds!=i,]))
		  X.lda.data <- X.app[folds!=i,]%*%X.lda$scaling
		  X.lda.testfold <- X.app[folds==i,]%*%X.lda$scaling


		# ---------- FDA -----------------------
		data.train = X.lda.data
		data.test = X.lda.testfold
		y.train = y.app[folds!=i]
		y.testfold = y.app[folds==i]
		ordre = c(1:dim(data.train)[2])


			for(k in 1:6){
				response <- rep(0,length(y.train))
				for(j in 1:length(y.train)){
					if(y.train[j]==k){
						response[j]=1
					}
				}
				newDataSet <- data.frame(data.train,response)
				formule = getFormulas(colnames(newDataSet), ordre,"response")
				if(k==1)
				{
					neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneBP1,neurone2),algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn1 <- ifelse(neuralnet1$net.result[[1]]>0.5,1,0)

				}
				else if( k==2)
				{
					neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneBP1,neurone2),algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn2 <- ifelse(neuralnet2$net.result[[1]]>0.5,1,0)

				}
				else if(k==3){
					neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneBP1,neurone2),algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn3 <- ifelse(neuralnet3$net.result[[1]]>0.5,1,0)

				}
				else if(k==4){
					neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneBP1,neurone2),algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn4 <- ifelse(neuralnet4$net.result[[1]]>0.5,1,0)

				}
				else if(k==5){
					neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneBP1,neurone2),algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn5 <- ifelse(neuralnet5$net.result[[1]]>0.5,1,0)

				}
				else if(k==6){
					neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneBP1,neurone2),algorithm="backprop",learningrate=0.01,err.fct="ce",linear.output=FALSE)
					nn6 <- ifelse(neuralnet6$net.result[[1]]>0.5,1,0)

				}
				
			}

			# predictions 

			if(!is.null(neuralnet1$net.result)){
				c1<-compute(neuralnet1,as.matrix(data.test))
			} else {c1$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet2$net.result)){
				c2<-compute(neuralnet2,as.matrix(data.test))
			} else {c2$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet3$net.result)){
				c3<-compute(neuralnet3,as.matrix(data.test))
			} else {c3$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet4$net.result)){
				c4<-compute(neuralnet4,as.matrix(data.test))
			} else {c4$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet5$net.result)){
				c5<-compute(neuralnet5,as.matrix(data.test))
			} else {c5$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet6$net.result)){
				c6<-compute(neuralnet6,as.matrix(data.test))
			} else {c6$net.result<-rep(0,dim(data.test)[1])}

			neuralnet.result <- rep(0,dim(data.test)[1])

			for(j in 1:dim(data.test)[1]){
				index <- which.max(c(c1$net.result[j],c2$net.result[j],c3$net.result[j],c4$net.result[j],c5$net.result[j],c6$net.result[j]))
				neuralnet.result[j] <- index
			}
			neuralnet.perf <- table(neuralnet.result,y.testfold)
			fda.error.neuralnetBP2[neurone2-2] <- fda.error.neuralnetBP2[neurone2-2]+  1-sum(diag(neuralnet.perf))/length(y.testfold)



		# ------------- FDA sans backprop --------------
		print("sans backprog")
		data.train = X.lda.transform[folds!=i,]
		data.test = X.lda.transform[folds==i,]
		y.train = y.app[folds!=i]
		y.testfold = y.app[folds==i]
		ordre = c(1:dim(data.train)[2])


			for(k in 1:6){
				response <- rep(0,length(y.train))
				for(j in 1:length(y.train)){
					if(y.train[j]==k){
						response[j]=1
					}
				}
				newDataSet <- data.frame(data.train,response)
				formule = getFormulas(colnames(newDataSet), ordre,"response")
				if(k==1)
				{
					neuralnet1 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneNoBP1,neurone2), err.fct="ce",linear.output=FALSE)
					nn1 <- ifelse(neuralnet1$net.result[[1]]>0.5,1,0)

				}
				else if( k==2)
				{
					neuralnet2 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneNoBP1,neurone2),err.fct="ce",linear.output=FALSE)
					nn2 <- ifelse(neuralnet2$net.result[[1]]>0.5,1,0)

				}
				else if(k==3){
					neuralnet3 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneNoBP1,neurone2),err.fct="ce",linear.output=FALSE)
					nn3 <- ifelse(neuralnet3$net.result[[1]]>0.5,1,0)

				}
				else if(k==4){
					neuralnet4 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneNoBP1,neurone2),err.fct="ce",linear.output=FALSE)
					nn4 <- ifelse(neuralnet4$net.result[[1]]>0.5,1,0)

				}
				else if(k==5){
					neuralnet5 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneNoBP1,neurone2),err.fct="ce",linear.output=FALSE)
					nn5 <- ifelse(neuralnet5$net.result[[1]]>0.5,1,0)

				}
				else if(k==6){
					neuralnet6 <- neuralnet(formule[dim(data.train)[2]],newDataSet,hidden=c(neuroneNoBP1,neurone2),err.fct="ce",linear.output=FALSE)
					nn6 <- ifelse(neuralnet6$net.result[[1]]>0.5,1,0)

				}
				
			}
			if(!is.null(neuralnet1$net.result)){
				c1<-compute(neuralnet1,as.matrix(data.test))
			} else {c1$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet2$net.result)){
				c2<-compute(neuralnet2,as.matrix(data.test))
			} else {c2$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet3$net.result)){
				c3<-compute(neuralnet3,as.matrix(data.test))
			} else {c3$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet4$net.result)){
				c4<-compute(neuralnet4,as.matrix(data.test))
			} else {c4$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet5$net.result)){
				c5<-compute(neuralnet5,as.matrix(data.test))
			} else {c5$net.result<-rep(0,dim(data.test)[1])}
			if(!is.null(neuralnet6$net.result)){
				c6<-compute(neuralnet6,as.matrix(data.test))
			} else {c6$net.result<-rep(0,dim(data.test)[1])}

			neuralnet.result <- rep(0,dim(data.test)[1])

			for(j in 1:dim(data.test)[1]){
				index <- which.max(c(c1$net.result[j],c2$net.result[j],c3$net.result[j],c4$net.result[j],c5$net.result[j],c6$net.result[j]))
				neuralnet.result[j] <- index
			}
			neuralnet.perf <- table(neuralnet.result,y.testfold)
			fda.error.neuralnetNoBP2[neurone2-2] <-fda.error.neuralnetNoBP2[neurone2-2]+ 1-sum(diag(neuralnet.perf))/length(y.testfold)
		}

}
#fda.error.neuralnetNoBP2<- (fda.error.neuralnetNoBP2/K)*100
fda.error.neuralnetBP2<- (fda.error.neuralnetBP2/K)*100

fda.error.neuralnetNoBP2<- c(8.730250142,10.444535856,9.259877378,7.972341384,10.420559832,9.243070655,10.471998590)

	neuroneNoBP2<- which.min(fda.error.neuralnetNoBP2)+2
	neuroneBP2<-which.min(fda.error.neuralnetBP2)+2

	cat("[NEURALNET] Neurone 2 optimal BP : ",neuroneBP2,", avec une erreur de : ",fda.error.neuralnetBP2[neuroneBP2-2])
	cat("[NEURALNET] Neurone 2 optimal No BP : ",neuroneNoBP2,", avec une erreur de : ",fda.error.neuralnetNoBP2[neuroneNoBP2-2])

# résultats :
# -LDA + BP : 28.66541302% avec 8 neurones
# -LDA : 7.972341384 avec 6 neurones

min=0
max=40
plot(c(3:9),fda.error.neuralnetBP2[1:7],type="l", ylim=c(min,max), ylab="", xlab="",col='blue')
par(new=TRUE)
plot(c(3:9),fda.error.neuralnetNoBP2[1:7],type="l", ylim=c(min,max), ylab="", xlab="",col='green')
# les meilleurs résultats sont pour size = 9 et le FDA. On a fait tourner une boucle de 5 a 15 noeuds cachés




# ----------- deuxieme essai avec une autre méthode par neuralnet et biclass ---------------- 

# cette fois ci on essaie pas juste le acp1 et on se rend compte que le fda fait de bien meilleurs résultats, on laisse tombé d'autant plus que les resultats sont limités de par le nombre de variables autorisés a prendre
# on obtient une erreur recurente lorsuq"on lance trop de fois neuralnet a la suite avec des jeux de valuers qui diffrents


# Nous choisissons donc de prendre 5 neuronnes cachés et nous avons autour de 6.5% d erreur
# on essaye une deuxieme couche de neuronnes cachés, on est sur du 6% d erreur avec du 3,3, ou du 4,3 c'est pareil

# couche de neuronnes cachés :
# nous avons lancé plusieurs fois une boucle pour connaitre le nombre de neuronnes optimal a garder lorsuq'on considere norte reseau a 
# une seule couche de neuronne caché : les meilleurs résultats etaient lorsuq'on gardait entre 3 et 5 neurones.
# avec 4 on est : 8.1% - 6.8% - 11.6% - 9.5% - 8.4%
# avec 3 on est : 7.5% - 5.8% - 6.1% - 6.5% - 6.7%
# avec 5 on est : 8.8% - 9.7% - 6.4% - 7.3% - 10%
# avec 2 on est : 6.16% - 5.7% - 8.8% - 7.7% - 6.3%
#  on remarque qu'en moyenne les résultats sont meilleurs avec 3 neuronnes dans la premiere couche cachée

# nous avons donc décidé d'essayer de rajouter une deuxieme couche de neuronnes cachés pour observer des changement ou une amélioratin de résultat qi poqqible
# avec (3,2) on est : 5% - 12% - 8.3% - 6.6% - 10% - 7.3%
# avec (3,3) on est : 7.8% - 8.7% - 6.7% - 9.4% - 9.15%
# avec (3,4) on est : 8.7% - 8.3% - 8.6% - 11.2% - 9.9%

# d'apres les resultats suiovant nous décidons de prendre une couche cachée constituée de 3neuronnes


# -------------------lors de ce dernier essai on va tenter de prendre l'algorithme de backpropagation avec la méthode la plus performante --> FDA

# premiere iteration 6.5% - 4.6% - 10.13% - 6.8% - 5.77% - 8.7%


# avec back : 8.7  -  7.4% -- 7.1% -- 6% -- 8.2%
# sans back : 8.7  -  6.2% -- 7.7% -- 4.1% -- 7.5%

# on remarque donc que l algorithme sans backpropagation est plus performant.

# Nous retznons donc le neuralnet avec 3 neuronnes cachés et sans algo de backprop

