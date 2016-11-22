library("FNN", character.only = TRUE)


phoneme = read.table("data/phoneme.data.txt",header=T,sep=",")

# --------------------------------------------- SCALING DATAS ---------------------------------------------------------


phoneme[,2:257] <- as.data.frame(scale(phoneme[,2:257]))
 
# -------------------------------------------- DATA SEPARATION --------------------------------------------------------

#	Nous utiliserons ici le prinicipe de cross validation :
#	Pour cela nous allons commencer par séparer les valeurs en 2 sous ensembles
#	Un ensemble de validation de taille 1/3 des données
#	Une ensemble d'apprentissage de taille 2/3 des données

#	L'ensemle d'apprentissage va etre divisé aléatoirement en sections
#	Nous choisissons arbitrairement de découper en 10 sections de 300 données chacunes
#	Sur ces 10 sections nous choisissons tour à tour 1 section qui representera 
#	notre ensemble de test. Nous allons donc effectuer la construction de notre modele 10 fois
#	et ensuite choisir celui qui aura les meilleures performances
#	Cette méthode nous permet de nous afrfanchir des risques de divergence de modele selon la prise aleatoire d un 
#	ensemble de test et d apprentissage.

data.dim <- dim(phoneme)

# 1 - Separation des donnees en test, app
napp <- round(2/3*data.dim[1])
indice <- sample(1:data.dim[1], napp)

cv.phoneme.train.data <- phoneme[indice,2:257]
cv.phoneme.validation.data <- phoneme[-indice,2:257]

cv.phoneme.train.label<- phoneme[indice,258]
cv.phoneme.validation.label <- phoneme[-indice,258]

phoneme.naive.error <- rep(0,1)
phoneme.tree.error <- rep(0,1)
phoneme.lda.error <- rep(0,1)
phoneme.fda.lda.error <- rep(0,1)
phoneme.fda.qda.error <- rep(0,1)
phoneme.qda.error <- rep(0,1)
phoneme.fda.glmnet.error <- rep(0,1)
phoneme.glmnet.error <- rep(0,1)

K <- 5 # Nombre de sections dans notre ensemble d'apprentissage

reg.fit<- regsubsets(phoneme.train.label~.,data=phoneme.train.data,method='forward',nvmax=256)
summary.regsubsets <- summary(reg.fit)
summary.regsubsets.which<-summary.regsubsets$which #permet de savoir quels variables sont dans quels modeles. (il faut decaler de 2)
LDA_ERROR <- matrix(0,ncol=2,nrow=257)
QDA_ERROR <- matrix(0,ncol=2,nrow=257)
KNN_ERROR <- matrix(0,ncol=2,nrow=257)
TREE_ERROR <- matrix(0,ncol=2,nrow=257)
BAYES_ERROR <- matrix(0,ncol=2,nrow=257)
lda.min <- rep(100,K)
qda.min <- rep(100,K)
knn.min <- rep(100,K)
tree.min <- rep(100,K)
bayes.min <- rep(100,K)
lda.subset <- summary.regsubsets.which[2,3:257]
qda.subset <- summary.regsubsets.which[2,3:257]
knn.subset <- summary.regsubsets.which[2,3:257]
tree.subset <- summary.regsubsets.which[2,3:257]
bayes.subset <- summary.regsubsets.which[2,3:257]
k.opt <- 0

folds <- sample(1:K,dim(cv.phoneme.train.data)[1],replace=TRUE)
for(i in 1:K){
	print("on lance une nouvelle iteration")
	phoneme.train.data <- cv.phoneme.train.data[folds!=i,]
	phoneme.train.label <- cv.phoneme.train.label[folds!=i]
	phoneme.test.data <- cv.phoneme.train.data[folds==i,]
	phoneme.test.label <- cv.phoneme.train.label[folds==i]

	# --------------- LDA ----------------------
	#phoneme.lda <- lda(phoneme.train.label~.,data=phoneme.train.data)
	#phoneme.lda.pred <- predict(phoneme.lda, newdata=phoneme.test.data)
	#phoneme.lda.perf <- table(phoneme.test.label,phoneme.lda.pred$class)
	#print((1 - sum(diag(phoneme.lda.perf))/(nrow(phoneme.test.data))))
	#phoneme.lda.error <- phoneme.lda.error + (1 - sum(diag(phoneme.lda.perf))/(nrow(phoneme.test.data)))

	# --------------- QDA ----------------------
	#phoneme.qda <- qda(phoneme.train.label~.,data=phoneme.train.data)
	#phoneme.qda.pred <- predict(phoneme.qda, newdata=phoneme.test.data)
	#phoneme.qda.perf <- table(phoneme.test.label,phoneme.qda.pred$class)
	#print(1 - sum(diag(phoneme.qda.perf))/(nrow(phoneme.test.data)))
	#phoneme.qda.error <- phoneme.qda.error + 1 - sum(diag(phoneme.qda.perf))/(nrow(phoneme.test.data))

	#		--- KPPV - 9.27% d'erreur - koptimal 8 ---
	#phoneme.knn.error<-rep(0,20)
	#for(k in 8:8)
	#{
	#	phoneme.knn <- knn(phoneme.train.data, phoneme.test.data, phoneme.train.label,k=k)
	#	phoneme.knn.error[k] <-phoneme.knn.error[k]  + (length(which(FALSE==(phoneme.knn==phoneme.test.label))))/length(phoneme.test.label)

	#}

	#phoneme.glmnet <- glmnet(as.matrix(phoneme.train.data),y=phoneme.train.label,family="multinomial")
	#phoneme.glmnet.pred <- predict(phoneme.glmnet,newx=as.matrix(phoneme.test.data),type="response",s=phoneme.glmnet$lambda.min)
	#phoneme.glmnet.res<-c(rep(0,dim(phoneme.test.data)[1]))
	#for (h in 1:dim(phoneme.test.data)[1])
	#{
	#	class <- ""
	#	res<-which.max(phoneme.glmnet.pred[h,1:5,100])
	#	{
	#		if(res==1)
	#		{
	#			class <- "aa"
	#		}
	#		else if(res==2){
	#			class <- "ao"
	#		}
	#		else if(res==3){
	#			class <- "dcl"
	#		}
	#		else if(res==4){
	#			class <- "iy"
	#		}
	#		else{
	#			class <- "sh"
	#		}
	#	}
	#	phoneme.glmnet.res[h] <- class 
	#}
	#phoneme.glmnet.perf <- table(phoneme.test.label,phoneme.glmnet.res)
	#print(1 - sum(diag(phoneme.glmnet.perf))/(nrow(phoneme.test.data)))
	#phoneme.glmnet.error <-phoneme.glmnet.error + 1 - sum(diag(phoneme.glmnet.perf))/(nrow(phoneme.test.data))

	#		--- Classifiation tree ------------
	#phoneme.tree<- tree(phoneme.train.label~ ., data=phoneme.train.data) 
	#phoneme.tree.pred<-predict(phoneme.tree, phoneme.test.data, type="class")
	#phoneme.tree.perf <- table(phoneme.tree.pred, phoneme.test.label)
	#print((sum(phoneme.tree.perf)-sum(diag(phoneme.tree.perf)))/nrow(phoneme.test.data))
	#phoneme.tree.error <- phoneme.tree.error + (sum(phoneme.tree.perf)-sum(diag(phoneme.tree.perf)))/nrow(phoneme.test.data)


	#		--- Classifieur bayesien naif ----
	#phoneme.naive<- naiveBayes(phoneme.train.label~., data=phoneme.train.data)
	#phoneme.naive.pred<-predict(phoneme.naive,newdata=phoneme.test.data)
	#phoneme.naive.perf <-table(phoneme.test.label,phoneme.naive.pred)
	#print(1-sum(diag(phoneme.naive.perf))/nrow(phoneme.test.data))
	#phoneme.naive.error <-phoneme.naive.error +  1-sum(diag(phoneme.naive.perf))/nrow(phoneme.test.data)

	#	--------------------FDA --------------------------
	#phoneme.fda.lda<-lda(phoneme.train.label~. ,data=phoneme.train.data)
	#U <- phoneme.fda.lda$scaling
	#X <- as.matrix(phoneme.train.data)
	#Z <- X%*%U

	#phoneme.fda.lda.test<-lda(phoneme.test.label~. ,data=phoneme.test.data)
	#Utest <- phoneme.fda.lda.test$scaling
	#Xtest <- as.matrix(phoneme.test.data)
	#Ztest <- Xtest%*%Utest

	#cp1 <- 1
	#cp2 <- 2

	#phoneme.fda.lda <- lda(phoneme.train.label~.,data=as.data.frame(Z))
	#phoneme.fda.lda.pred <- predict(phoneme.fda.lda, newdata=as.data.frame(Ztest))
	#phoneme.fda.lda.perf <- table(phoneme.test.label,phoneme.fda.lda.pred$class)
	#print(1 - sum(diag(phoneme.fda.lda.perf))/(nrow(phoneme.test.data)))
	#print(phoneme.fda.lda.perf)
	#phoneme.fda.lda.error <-phoneme.fda.lda.error + (1 - sum(diag(phoneme.fda.lda.perf))/(nrow(phoneme.test.data)))

	#phoneme.fda.qda <- qda(phoneme.train.label~.,data=as.data.frame(Z))
	#phoneme.fda.qda.pred <- predict(phoneme.fda.qda, newdata=as.data.frame(Ztest))
	#phoneme.fda.qda.perf <- table(phoneme.test.label,phoneme.fda.qda.pred$class)
	#print(1 - sum(diag(phoneme.fda.qda.perf))/(nrow(phoneme.test.data)))
	#print(phoneme.fda.qda.perf)
	#phoneme.fda.qda.error <- phoneme.fda.qda.error + 1 - sum(diag(phoneme.fda.qda.perf))/(nrow(phoneme.test.data))

	#phoneme.fda.glmnet <- glmnet(as.matrix(Z),y=phoneme.train.label,family="multinomial")
	#phoneme.fda.glmnet.pred <- predict(phoneme.fda.glmnet,newx=as.matrix(Ztest),type="response",s=phoneme.fda.glmnet$lambda.min)
	#phoneme.fda.glmnet.res<-c(rep(0,dim(phoneme.test.data)[1]))
	#for (h in 1:dim(phoneme.test.data)[1])
	#{
	#	class <- ""
	#	res<-which.max(phoneme.fda.glmnet.pred[h,1:5,100])
	#	{
	#		if(res==1)
	#		{
	#			class <- "aa"
	#		}
	#		else if(res==2){
	#			class <- "ao"
	#		}
	#		else if(res==3){
	#			class <- "dcl"
	#		}
	#		else if(res==4){
	#			class <- "iy"
	#		}
	#		else{
	#			class <- "sh"
	#		}
	#	}
	#	phoneme.fda.glmnet.res[h] <- class 
	#}
	#phoneme.fda.glmnet.perf <- table(phoneme.test.label,phoneme.fda.glmnet.res)
	#print(phoneme.fda.glmnet.perf)
	#print(1 - sum(diag(phoneme.fda.glmnet.perf))/(nrow(phoneme.test.data)))
	#phoneme.fda.glmnet.error <-phoneme.fda.glmnet.error + 1 - sum(diag(phoneme.fda.glmnet.perf))/(nrow(phoneme.test.data))

	# ------------------------ subset selection --------------------

	print("Reduction du nombre de variable en utilisant la subset selection")

	for(l in 10:256)#ca sert a rien de le faire jusqu'a 256 on a deja les resultats plus haut.
	{
		print(l)
		# selection des nouveaux jeux de données selon le nombre de variables gardés.
		new.phoneme.train.data<-phoneme.train.data[,summary.regsubsets.which[l,3:257]]
		new.phoneme.train.data<-as.data.frame(new.phoneme.train.data)
		new.phoneme.test.data<-phoneme.test.data[,summary.regsubsets.which[l,3:257]]
		new.phoneme.test.data<-as.data.frame(new.phoneme.test.data)

		#calcul des nouveaux taux d'erreur de chaque modele

		#  		--- LDA - 7.87% d erreur - 132 variables gardées ---
		new.phoneme.lda <- lda(phoneme.train.label~.,data=new.phoneme.train.data)
		new.phoneme.lda.pred <- predict(new.phoneme.lda, newdata=new.phoneme.test.data)
		new.phoneme.lda.perf <- table(phoneme.test.label,new.phoneme.lda.pred$class)
		LDA_ERROR[l,2] <- 1 - sum(diag(new.phoneme.lda.perf))/(nrow(phoneme.test.data))
		LDA_ERROR[l,1] <- l
		if(LDA_ERROR[l,2]<lda.min[i])
		{
			lda.min[i] <- LDA_ERROR[l,2]
			lda.subset <- summary.regsubsets.which[l,3:257]
		}
		#		--- QDA - 7.8% d erreur - 37 variables gardées ---
		new.phoneme.qda <- qda(phoneme.train.label~.,data=new.phoneme.train.data)
		new.phoneme.qda.pred <- predict(new.phoneme.qda, newdata=new.phoneme.test.data)
		new.phoneme.qda.perf <- table(phoneme.test.label,new.phoneme.qda.pred$class)
		QDA_ERROR[l,2] <- 1 - sum(diag(new.phoneme.qda.perf))/(nrow(phoneme.test.data))
		QDA_ERROR[l,1] <- l
		if(QDA_ERROR[l,2]<qda.min[i])
		{
			qda.min[i] <- QDA_ERROR[l,2]
			qda.subset <- summary.regsubsets.which[l,3:257]
		}
		#		--- KNN - 7.87% d erreur - k optimal 8 - 48 variables gardées ---
		for(k in 8:8)
		{
			new.phoneme.knn <- knn(new.phoneme.train.data, new.phoneme.test.data, phoneme.train.label,k=k)
			KNN_ERROR[l,2] <- (length(which(FALSE==(new.phoneme.knn==phoneme.test.label))))/length(phoneme.test.label)
			KNN_ERROR[l,1] <- l
			if(KNN_ERROR[l,2]<knn.min[i])
			{
				knn.min[i] <- KNN_ERROR[l,2]
				knn.subset <- summary.regsubsets.which[l,3:257]
				k.opt <- k
			}
		}

		#		--- Classifiation tree - 12.53% d'erreur avec 60 variables ---
		new.phoneme.tree<- tree(phoneme.train.label~ ., data=new.phoneme.train.data) 
		new.phoneme.tree.pred<-predict(new.phoneme.tree, new.phoneme.test.data, type="class")
		new.phoneme.tree.perf <- table(new.phoneme.tree.pred, phoneme.test.label)
		TREE_ERROR[l,2] <- (sum(new.phoneme.tree.perf)-sum(diag(new.phoneme.tree.perf)))/nrow(phoneme.test.data)
		TREE_ERROR[l,1] <- l
		if(TREE_ERROR[l,2]<tree.min[i])
		{
			tree.min[i] <- TREE_ERROR[l,2]
			tree.subset <- summary.regsubsets.which[l,3:257]
		}


		#		--- Classifieur bayesien naif - 10.27% d'erreur avec 57 variables ---
		new.phoneme.naive<- naiveBayes(phoneme.train.label~., data=new.phoneme.train.data)
		new.phoneme.naive.pred<-predict(new.phoneme.naive,newdata=new.phoneme.test.data)
		new.phoneme.naive.perf <-table(phoneme.test.label,new.phoneme.naive.pred)
		new.phoneme.naive.error <- 1-sum(diag(new.phoneme.naive.perf))/nrow(phoneme.test)
		BAYES_ERROR[l,2] <- 1-sum(diag(new.phoneme.naive.perf))/nrow(phoneme.test.data)
		BAYES_ERROR[l,1] <- l
		if(BAYES_ERROR[l,2]<bayes.min[i])
		{
			bayes.min[i] <- BAYES_ERROR[l,2]
			bayes.subset <- summary.regsubsets.which[l,3:257]
		}
		
	}
	print("Apres subset selection : ")
	print("LDA error minimale : ")
	print(lda.min)
	plot(LDA_ERROR)
	print("QDA error minimale : ")
	print(qda.min)
	plot(QDA_ERROR)
	print("KNN error minimale : ")
	print(knn.min)
	plot(KNN_ERROR)
	print("TREE error minimale : ")
	print(tree.min)
	plot(TREE_ERROR)
	print("BAYES error minimale : ")
	print(bayes.min)
	plot(BAYES_ERROR)

	
}
print("RESULTATS FINAUX")
phoneme.lda.error<-phoneme.lda.error/K
phoneme.qda.error <- phoneme.qda.error/K
phoneme.glmnet.error <- phoneme.glmnet.error/K
phoneme.tree.error<-phoneme.tree.error/K
phoneme.naive.error<-phoneme.naive.error/K
phoneme.fda.lda.error <- phoneme.fda.lda.error/K
phoneme.fda.qda.error <- phoneme.fda.qda.error/K
phoneme.fda.glmnet.error <- phoneme.fda.glmnet.error/K
phoneme.knn.error <- phoneme.knn.error/K
print("LDA")
print(phoneme.lda.error)
print("QDA")
print(phoneme.qda.error)
print("Regression logistique")
print(phoneme.glmnet.error)
print("TREE")
print(phoneme.tree.error)
print("NAIVE")
print(phoneme.naive.error)
print("FDA + LDA")
print(phoneme.fda.lda.error)
print("FDA + QDA")
print(phoneme.fda.qda.error)
print("FDA + Regression logistique")
print(phoneme.fda.glmnet.error)
print("KNN")
print(phoneme.knn.error)

# LDA : 0.06844741 0.07540984 0.04841402 0.06303237 0.07528642 mean : 0.06611801
# QDA : 0.08347245 0.07377049 0.06677796 0.08006814 0.08346972 mean : 0.07751175
# KNN : 0.08514190 0.07377049 0.06010017 0.08006814 0.07855974 mean : 0.07552809
# TREE : 0.1402337 0.1491803 0.1101836 0.1362862 0.1243863 mean : 0.132054
# BAYES : 0.10016694 0.09344262 0.06844741 0.09028961 0.08510638 mean : 0.08749059



#	---------------------------------------------- INTERPRETATION  -------------------------------------------------------------

# La FDA ne marche pas avec la cross validation, il y a trop d'erreur qui sont influencé par le choix de l'enemble d'apprentissage ce qui donne un mauvais clustering des données
# Le hoix du modele classique LDA est assez performant 
# Lorsqu'on lance une subset selection sur un nombre de variable , on va pouvoir voir si ça amélire nos résultats

# La cross validation nous donne le rsultat suivant : le meilleur resultat est obtenu en utilisant une analyse discriminante linéaire.
# Nous allons maintenant valider ce modele en l'appliquant à notre ensemble de validation pour connaitre le vrai taux d'erreur

# Pour cela on reconstruit notre ensemble d'apprentissage en entier. 
# on lui applique le modele retenu
# on teste sur notre ensmble de validation

cv.phoneme.lda <- lda(cv.phoneme.train.label~.,data=cv.phoneme.train.data)
cv.phoneme.lda.pred <- predict(cv.phoneme.lda, newdata=cv.phoneme.validation.data)
cv.phoneme.lda.perf <- table(cv.phoneme.validation.label,cv.phoneme.lda.pred$class)
print((1 - sum(diag(cv.phoneme.lda.perf))/(nrow(cv.phoneme.validation.data))))
cv.phoneme.lda.error <- (1 - sum(diag(cv.phoneme.lda.perf))/(nrow(cv.phoneme.validation.data)))

