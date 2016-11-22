# LOAD DATA

library(MASS)
library(pROC)
library(nnet)
library(glmnet)
library(leaps)
library("FNN", character.only = TRUE)
library("hydroGOF", character.only = TRUE)
library(tree)
library(e1071)
phoneme = read.table("data/phoneme.data.txt",header=T,sep=",")
# ====
# x.1, x.2, ... x.256, g, speaker
# ====
x <- phoneme$g

y <- phoneme$speaker

# Pour que tu ne sois pas paumé et que tu comprennes ce que j'ai fait, je laisse dans le script pour le moment
# et il te suffit de print a chaque fois

# 1ere question : devons nous entrainer notre modele a reconnaitre les mots d'un certain locuteur 
# ou devons nous reussir a reconnaitre pour n'importe qui et c'est la reconnaissance du mot qui
# est importante ?
#		Marceau : De mon point de vue, ce jeu de données est fait pour aider a la reconnaissance de discours ("from the TIMIT database 
#		which is a widely used resource for research in speech recognition"). On entend par la que les personnes qui veulent creer des programmes de reconnaissances vocales utilisent cette base de donnees
#		En fouillant un peu sur internet on peut se rendre copte assez facilement qu'au final on a aucune info sur ces speakers. Je penche vers la version suivante :
#		Nous devons entrainer notre modele pour reussir a reconnaitre pour n'importe qui et c'est la reconnaissance du mot qui
# 		est importante

# cette prise de position est primordiale, car elle va notamment impliquer une difference au niveau
# de notre division des datas ( test et apprentissage)
# - si on veut bien reconnaitre que les personnes presentes dans nos datas alors il faut diviser notre
# ensemble de maniere a avoir 2/3 des datas d'une personnes dans l'apprentissage et 1/3 de
# ses enregistrements dans le test.
# - si on veut reconnaitre maintenant avant tous l'utilisation des mots dans le discours, 
# alors il est plus interessant de garder 2/3 des personnes et donc tout leur enregistrements
# dans les donnees d'apprentissage et de garder 1/3 des personnes inconnus par notre modele.
#		Marceau : Je penche donc vers ce deuxieme modele. J'aurai cependant une petite remarque a faire la dessus : comme nous cherchons ici a reconnaitre 
#		seuleemnt certains mots (sh, dcl ...), le but de notre modele sera de dire : tiens le mot que je viens d'entendre c'est "sh".
#		Il serait peut etre plus judicieux de prendre 2/3 des enregistrements de chaque mot à reconnaitre.

# === Description de nos datas ===
# je t'invite deja a lire et relire le doc phoneme.info.txt qui nous ait fournis
# j'avoue que j'ai mis du temps a comprendre mais voila ce que j'ai retenu sur nos datas :
# on fait de la reconnaissance vocale sur des discours, ce que l'on veut retenir nous
# ce n'est que 5 phonemes 'aa', 'sh', 'dck', 'ao', et 'iy' (contrairement a la vraie reconnaissance vocale ou tu as
# beaucoup plus de phonemes a prendre en compte )

# Pour pouvoir creer un modele de reconnaissance de ces phonemes, on a eu tout d'abord
# 50 hommes qui ont fait des discours, on en a retire 4509 petits bouts de discours de 32ms
# avec environ 2 examples de phonemes par speaker.
# 1 frame ( petit bouts de discours ) = 1 phoneme = 512 samples ( on le divise en 512 petits bouts again)

# Voila la repartition de nos differents frames
# aa   ao dcl   iy  sh 
# 695 1022 757 1163 872

# Pour chacun de nos frames, on fait un log-periodogram, technique utilisé dans la reconnaissance vocale
# et ce log-periodogram comprend 256 features
# nos datas 
# --------------------------------------------- SCALING DATAS ---------------------------------------------------------
print("scaling datas")
# Toutes les datas ont plus ou moins une variance et une une moyenne identique, nous centrons et reduisons quand meme 
# pour etre sur de rendre ces variables comparables

phoneme[,2:257] <- as.data.frame(scale(phoneme[,2:257]))


# -------------------------------------------- DATA SEPARATION --------------------------------------------------------
print("data separation")
#		Marceau : Je propose la dessus de respecter les 5 criteres de separation donnes par TIMIT
#			1 : 2/3 de train et 1/3 de test
#			2 : Aucun speaker ne doit apparaitre dans train et test (il suffit de faire attention pour cela a bien basculer tous les enregistrements
#				par speakers quand on fait un ensemble)
#			3 : Toutes les regions doivent apparaitre dans nos ensembles de test/train, avec au moins un speaker de chaque sexe (a verifier a la main)
#			4 : Tous les phonemes doivent apparaitre (ici on fait justement attention a ca, en prenant la meme proportion de chaque phonemes)

# J'ai analysé chaque jeu de donnees suivant le phoneme, ca m'a permis de creer a la main les ensembles d'apprentissage et de test en faisant attention
# aux criteres annoces ci dessus. On a bien des ensembles avec 2/3 de train et 1/3 de test, aucun speaker n'apparait dans le test et dans le train. 
# les speaker de chaque region + pour chaque sexe + pour chaque phoneme sont presents dans nos ensembles de test ET train

#		--- creation de l'ensemble pour les aa ---
aa <- subset(phoneme,g=='aa')
#il faut en rajouter 56
aa.train <- rbind(aa[1:450,],aa[495:507,])
aa.test <- rbind(aa[451:494,],aa[508:695,])

#		--- creation de l'ensemble pour les sh ---
sh <- subset(phoneme,g=='sh')
#il faut en rajouter 67
sh.train <- rbind(sh[1:569,],sh[621:635,])
sh.test <- rbind(sh[570:622,],sh[636:872,])

#		--- creation de l'ensemble pour les dcl ---
dcl <- subset(phoneme,g=='dcl')
#il faut en rajouter 57
dcl.train <- rbind(dcl[1:493,],dcl[540:552,])
dcl.test <- rbind(dcl[494:539,],dcl[553:757,])

#		--- creation de l'ensemble pour les ao ---
ao <- subset(phoneme,g=='ao')
#il en manque 77
ao.train <- rbind(ao[1:667,],ao[730:744,])
ao.test <- rbind(ao[668:729,],ao[745:1022,])

#		--- creation de l'ensemble pour les iy ---
iy <- subset(phoneme,g=='iy')
#il en manque 77
iy.train <- rbind(iy[1:756,],iy[811:832,])
iy.test <- rbind(iy[757:812,],iy[833:1163,])

#		--- assemblage de tous les dataframes ---
phoneme.train <- rbind(aa.train, sh.train, ao.train, dcl.train, iy.train)
phoneme.train.data <- phoneme.train[,2:257]
phoneme.train.label <- phoneme.train$g
phoneme.test <- rbind(aa.test, sh.test, ao.test, dcl.test, iy.test)
phoneme.test.data <- phoneme.test[,2:257]
phoneme.test.label <- phoneme.test[,258]


# ------------------------------------------------ CONSTRUCTION OF MODELS -------------------------------------------------
print("Prédiction des modeles sans travail préalable des données")
#on doit ici predire une classe et non une valeur, on utilisera donc des methodes de classification et non de regression

#  		--- LDA - 8.6% d erreur ---
print("LDA - Error : ")
phoneme.lda <- lda(phoneme.train.label~.,data=phoneme.train.data)
phoneme.lda.pred <- predict(phoneme.lda, newdata=phoneme.test.data)
phoneme.lda.perf <- table(phoneme.test.label,phoneme.lda.pred$class)
phoneme.lda.error <- 1 - sum(diag(phoneme.lda.perf))/(nrow(phoneme.test))
print(phoneme.lda.error)

#		--- QDA - 18.93% d erreur ---
print("QDA - Error : ")
phoneme.qda <- qda(phoneme.train.label~.,data=phoneme.train.data)
phoneme.qda.pred <- predict(phoneme.qda, newdata=phoneme.test.data)
phoneme.qda.perf <- table(phoneme.test.label,phoneme.qda.pred$class)
phoneme.qda.error <- 1 - sum(diag(phoneme.qda.perf))/(nrow(phoneme.test))
print(phoneme.qda.error)


#		--- Regression logistique - 11.13% d'erreur ---
print("Regression logistique - Error : ")
phoneme.glmnet <- glmnet(as.matrix(phoneme.train.data),y=phoneme.train.label,family="multinomial")
phoneme.glmnet.pred <- predict(phoneme.glmnet,newx=as.matrix(phoneme.test.data),type="response",s=phoneme.glmnet$lambda.min)
# phoneme.glmnet.pred est un tableau 3 dimensions :
#	- La premiere est sur nos obsvervations (1500 dans l'ensemble de test)
#	- La deuxieme est sur nos types de phonemes (5types de phonemes)
#	- La troisieme est sur l'iteration
phoneme.glmnet.res<-c(rep(0,1500))
for (i in 1:1500)
{
	class <- ""
	res<-which.max(phoneme.glmnet.pred[i,1:5,100])
	{
		if(res==1)
		{
			class <- "aa"
		}
		else if(res==2){
			class <- "ao"
		}
		else if(res==3){
			class <- "dcl"
		}
		else if(res==4){
			class <- "iy"
		}
		else{
			class <- "sh"
		}
	}
	phoneme.glmnet.res[i] <- class 
}
phoneme.glmnet.perf <- table(phoneme.test.label,phoneme.glmnet.res)
phoneme.glmnet.error <- 1 - sum(diag(phoneme.glmnet.perf))/(nrow(phoneme.test))
print(phoneme.glmnet.error)

#		--- KPPV - 9.27% d'erreur - koptimal 8 ---
print("KNN - Errors : ")
phoneme.knn.error<-rep(0,20)
for(k in 8:8)
{
	phoneme.knn <- knn(phoneme.train.data, phoneme.test.data, phoneme.train.label,k=k)
	phoneme.knn.error[k] <- (length(which(FALSE==(phoneme.knn==phoneme.test.label))))/length(phoneme.test.label)

}
print(phoneme.knn.error)

#		--- Classifiation tree - 14.8% d'erreur ---
print("TREE - Errors : ")
phoneme.tree<- tree(phoneme.train.label~ ., data=phoneme.train.data) 
phoneme.tree.pred<-predict(phoneme.tree, phoneme.test.data, type="class")
phoneme.tree.perf <- table(phoneme.tree.pred, phoneme.test.label)
phoneme.tree.error <- (sum(phoneme.tree.perf)-sum(diag(phoneme.tree.perf)))/nrow(phoneme.test.data)
print(phoneme.tree.error)


#		--- Classifieur bayesien naif - 12.53% d'erreur ---
print("BAYES - Errors : ")
phoneme.naive<- naiveBayes(phoneme.train.label~., data=phoneme.train.data)
phoneme.naive.pred<-predict(phoneme.naive,newdata=phoneme.test.data)
phoneme.naive.perf <-table(phoneme.test.label,phoneme.naive.pred)
phoneme.naive.error <- 1-sum(diag(phoneme.naive.perf))/nrow(phoneme.test.data)
print(phoneme.naive.error)

#regarder si toutes les varibles sont importants (regarder la correlation des variables)
#regularisation : regarder si rajouter un terme on a pas un meilleur modele (regarder la correlation des variables)
#facteur analysis : changemet de repere 
#faire un ACP pour peut etre reduire


# ------------------------------------------------ SUBSET SELECTION -------------------------------------------------
print("Reduction du nombre de variable en utilisant la subset selection")
reg.fit<- regsubsets(phoneme.train.label~.,data=phoneme.train.data,method='forward',nvmax=256)
summary.regsubsets <- summary(reg.fit)
summary.regsubsets.which<-summary.regsubsets$which #permet de savoir quels variables sont dans quels modeles. (il faut decaler de 2)
LDA_ERROR <- matrix(0,ncol=2,nrow=250)
QDA_ERROR <- matrix(0,ncol=2,nrow=250)
KNN_ERROR <- matrix(0,ncol=2,nrow=250)
TREE_ERROR <- matrix(0,ncol=2,nrow=250)
BAYES_ERROR <- matrix(0,ncol=2,nrow=250)
lda.min <- 100
qda.min <- 100
knn.min <- 100
tree.min <- 100
bayes.min <- 100
lda.subset <- summary.regsubsets.which[2,3:257]
qda.subset <- summary.regsubsets.which[2,3:257]
knn.subset <- summary.regsubsets.which[2,3:257]
tree.subset <- summary.regsubsets.which[2,3:257]
bayes.subset <- summary.regsubsets.which[2,3:257]
k.opt <- 0
for(i in 132:132)#ca sert a rien de le faire jusqu'a 256 on a deja les resultats plus haut.
{
	print(i)
	# selection des nouveaux jeux de données selon le nombre de variables gardés.
	new.phoneme.train.data<-phoneme.train.data[,summary.regsubsets.which[i,3:257]]
	new.phoneme.train.data<-as.data.frame(new.phoneme.train.data)
	new.phoneme.test.data<-phoneme.test.data[,summary.regsubsets.which[i,3:257]]
	new.phoneme.test.data<-as.data.frame(new.phoneme.test.data)

	#calcul des nouveaux taux d'erreur de chaque modele

	#  		--- LDA - 7.87% d erreur - 132 variables gardées ---
	new.phoneme.lda <- lda(phoneme.train.label~.,data=new.phoneme.train.data)
	new.phoneme.lda.pred <- predict(new.phoneme.lda, newdata=new.phoneme.test.data)
	new.phoneme.lda.perf <- table(phoneme.test.label,new.phoneme.lda.pred$class)
	LDA_ERROR[i,2] <- 1 - sum(diag(new.phoneme.lda.perf))/(nrow(phoneme.test))
	LDA_ERROR[i,1] <- i
	if(LDA_ERROR[i,2]<lda.min)
	{
		lda.min <- LDA_ERROR[i,2]
		lda.subset <- summary.regsubsets.which[i,3:257]
	}

	#		--- QDA - 7.8% d erreur - 37 variables gardées ---
	new.phoneme.qda <- qda(phoneme.train.label~.,data=new.phoneme.train.data)
	new.phoneme.qda.pred <- predict(new.phoneme.qda, newdata=new.phoneme.test.data)
	new.phoneme.qda.perf <- table(phoneme.test.label,new.phoneme.qda.pred$class)
	QDA_ERROR[i,2] <- 1 - sum(diag(new.phoneme.qda.perf))/(nrow(phoneme.test))
	QDA_ERROR[i,1] <- i
	if(QDA_ERROR[i,2]<qda.min)
	{
		qda.min <- QDA_ERROR[i,2]
		qda.subset <- summary.regsubsets.which[i,3:257]
	}

	#		--- KNN - 7.87% d erreur - k optimal 8 - 48 variables gardées ---
	for(k in 8:8)
	{
		new.phoneme.knn <- knn(new.phoneme.train.data, new.phoneme.test.data, phoneme.train.label,k=k)
		KNN_ERROR[i,2] <- (length(which(FALSE==(new.phoneme.knn==phoneme.test.label))))/length(phoneme.test.label)
		KNN_ERROR[i,1] <- i
		if(KNN_ERROR[i,2]<knn.min)
		{
			knn.min <- KNN_ERROR[i,2]
			knn.subset <- summary.regsubsets.which[i,3:257]
			k.opt <- k
		}
	}

	#		--- Classifiation tree - 12.53% d'erreur avec 60 variables ---
	new.phoneme.tree<- tree(phoneme.train.label~ ., data=new.phoneme.train.data) 
	new.phoneme.tree.pred<-predict(new.phoneme.tree, new.phoneme.test.data, type="class")
	new.phoneme.tree.perf <- table(new.phoneme.tree.pred, phoneme.test.label)
	TREE_ERROR[i,2] <- (sum(new.phoneme.tree.perf)-sum(diag(new.phoneme.tree.perf)))/nrow(phoneme.test)
	TREE_ERROR[i,1] <- i
	if(TREE_ERROR[i,2]<tree.min)
	{
		tree.min <- TREE_ERROR[i,2]
		tree.subset <- summary.regsubsets.which[i,3:257]
	}


	#		--- Classifieur bayesien naif - 10.27% d'erreur avec 57 variables ---
	new.phoneme.naive<- naiveBayes(phoneme.train.label~., data=new.phoneme.train.data)
	new.phoneme.naive.pred<-predict(new.phoneme.naive,newdata=new.phoneme.test.data)
	new.phoneme.naive.perf <-table(phoneme.test.label,new.phoneme.naive.pred)
	new.phoneme.naive.error <- 1-sum(diag(new.phoneme.naive.perf))/nrow(phoneme.test)
	BAYES_ERROR[i,2] <- 1-sum(diag(new.phoneme.naive.perf))/nrow(phoneme.test)
	BAYES_ERROR[i,1] <- i
	if(BAYES_ERROR[i,2]<bayes.min)
	{
		bayes.min <- BAYES_ERROR[i,2]
		bayes.subset <- summary.regsubsets.which[i,3:257]
	}
	
}
print("Apres subset selection : ")
print("LDA error minimale : ")
print(lda.min)
print("QDA error minimale : ")
print(qda.min)
print("KNN error minimale : ")
print(knn.min)
print("TREE error minimale : ")
print(tree.min)
print("BAYES error minimale : ")
print(bayes.min)

# On peut deduire de ce modele qu'une selection de variable reduit notre erreur moyenne de nos modele.
# Pour l analyse discriminnate lineaire  l erreur descend a 7.87% lorsqu'on garde 132 variables
# Pour l analyse discriminante quadratique l erreur descend a 7.4% lorsqu on garde 37 variables
# Pour les KNN l erreur descend a 7.87% lorsqu'on garde 48 variables avec un k optimnal qui reste a 8

# ------------------------------------------------ REGRESSION RIDGE & LASSO -------------------------------------------------

# Permet de estimer un modele avec des variables fortement correlees
# Avantage de la regression ridge : les variables explicatrices tres correlees se combinent pour se renforcer mutuellement
# Cette methode garde toutes les variables mais aucun moyen de savoir lesquelles ont plus de poids que d'autres

# La methode LASSO met a 0 les variables peu explicatives
# Si les variables sont correlees, l'algorithme en choisi une et la met a 0

# ------------------------------------------------ ANALYSE EN COMPOSANTE PRINCIPALE -------------------------------------------------
print("ACP")
# Nous avons effectué une analyse en composante principale 
# l'idee serait de creer un nouvel axe factoriel et de creer un modele a partir de la
phoneme.acp <- princomp(phoneme.train.data)
phoneme.acp.train.scores <- as.data.frame(phoneme.acp$scores)
phoneme.acp.test <- princomp(phoneme.test.data)
phoneme.acp.test.scores <- as.data.frame(phoneme.acp.test$scores)
# Lorsqu'on regarde nos vecteurs propres, on remarque qu on peut en garder 2 ou 3 pour expliquer nos 257 variables de maniere efficace
plot(phoneme.acp$scores[1:dim(phoneme.acp$scores)[1],1:2], col=c('red','green','yellow','black','blue')[phoneme.train.label])
# Les aa(red) les sh(green) et les dcl(black) se ressemblent beaucoup avec une acp avec le premier et deuxieme axe factoriel
plot(phoneme.acp$scores[1:dim(phoneme.acp$scores)[1],2:3], col=c('red','green','yellow','black','blue')[phoneme.train.label])
# Les dcl se démarquent tres bien des aa et des sh avec le deuxieme et le troisieme axe factoriel
plot(phoneme.acp$scores[1:dim(phoneme.acp$scores)[1],1],phoneme.acp$scores[1:dim(phoneme.acp$scores)[1],3], col=c('red','green','yellow','black','blue')[phoneme.train.label])
# L utilisation des la derniere combinaison d'axes factoriels ne sufit pas à séparer aa et sh qui restent tres corrélés.


#  		--- LDA - 13% d erreur ---
phoneme.acp.lda <- lda(phoneme.train.label~.,data=phoneme.acp.train.scores[,1:5])
phoneme.acp.lda.pred <- predict(phoneme.acp.lda, newdata=phoneme.acp.test.scores[,1:5])
phoneme.acp.lda.perf <- table(phoneme.test.label,phoneme.acp.lda.pred$class)
phoneme.acp.lda.error <- 1 - sum(diag(phoneme.acp.lda.perf))/(nrow(phoneme.test))
print("LDA avec ACP 5 dimensions: ")
print(phoneme.acp.lda.error)

#		--- QDA - 10.87% d erreur ---
phoneme.acp.qda <- qda(phoneme.train.label~.,data=phoneme.acp.train.scores[,1:5])
phoneme.acp.qda.pred <- predict(phoneme.acp.qda, newdata=phoneme.acp.test.scores[,1:5])
phoneme.acp.qda.perf <- table(phoneme.test.label,phoneme.acp.qda.pred$class)
phoneme.acp.qda.error <- 1 - sum(diag(phoneme.acp.qda.perf))/(nrow(phoneme.test))
print("QDA avec ACP 5 dimensions: ")
print(phoneme.acp.qda.error)

#		--- Regression logistique - 11.07% d'erreur ---
phoneme.acp.glmnet <- glmnet(as.matrix(phoneme.acp.train.scores[,1:5]),y=phoneme.train.label,family="multinomial")
phoneme.acp.glmnet.pred <- predict(phoneme.acp.glmnet,newx=as.matrix(phoneme.acp.test.scores[,1:5]),type="response",s=phoneme.acp.glmnet$lambda.min)
# phoneme.glmnet.pred est un tableau 3 dimensions :
#	- La premiere est sur nos obsvervations (1500 dans l'ensemble de test)
#	- La deuxieme est sur nos types de phonemes (5types de phonemes)
#	- La troisieme est sur l'iteration
phoneme.acp.glmnet.res<-c(rep(0,1500))
for (i in 1:1500)
{
	class <- ""
	res<-which.max(phoneme.acp.glmnet.pred[i,1:5,100])
	{
		if(res==1)
		{
			class <- "aa"
		}
		else if(res==2){
			class <- "ao"
		}
		else if(res==3){
			class <- "dcl"
		}
		else if(res==4){
			class <- "iy"
		}
		else{
			class <- "sh"
		}
	}
	phoneme.acp.glmnet.res[i] <- class 
}
phoneme.acp.glmnet.perf <- table(phoneme.test.label,phoneme.acp.glmnet.res)
phoneme.acp.glmnet.error <- 1 - sum(diag(phoneme.acp.glmnet.perf))/(nrow(phoneme.test))
print("Regression logistique avec ACP 5 dimensions : ")
print(phoneme.acp.glmnet.error)

#		--- KNN - 11.2% d'erreur - koptimal 8 ---
phoneme.acp.knn.error<-rep(0,20)
for(k in 1:10)
{
	phoneme.acp.knn <- knn(phoneme.acp.train.scores[,1:5], phoneme.acp.test.scores[,1:5], phoneme.train.label,k=k)
	phoneme.acp.knn.error[k] <- (length(which(FALSE==(phoneme.acp.knn==phoneme.test.label))))/length(phoneme.test.label)

}
print("KNN avec ACP 5 dimensions: ")
print(phoneme.acp.knn.error)

#		--- Classifiation tree - 15.73% d'erreur ---
print("TREE - Errors : ")
phoneme.acp.tree<- tree(phoneme.train.label~ ., data=phoneme.acp.train.scores[,1:5]) 
phoneme.acp.tree.pred<-predict(phoneme.acp.tree, phoneme.acp.test.scores[,1:5], type="class")
phoneme.acp.tree.perf <- table(phoneme.acp.tree.pred, phoneme.test.label)
phoneme.acp.tree.error <- (sum(phoneme.acp.tree.perf)-sum(diag(phoneme.acp.tree.perf)))/nrow(phoneme.test.data)
print(phoneme.acp.tree.error)


#		--- Classifieur bayesien naif - 12.47% d'erreur ---
print("BAYES - Errors : ")
phoneme.acp.naive<- naiveBayes(phoneme.train.label~., data=phoneme.acp.train.scores[,1:5])
phoneme.acp.naive.pred<-predict(phoneme.acp.naive,newdata=phoneme.acp.test.scores[,1:5])
phoneme.acp.naive.perf <-table(phoneme.test.label,phoneme.acp.naive.pred)
phoneme.acp.naive.error <- 1-sum(diag(phoneme.acp.naive.perf))/nrow(phoneme.test.data)
print(phoneme.acp.naive.error)
# On peut donc déduire de cette analyse 2 choses 
#	- Nos deux phonemes a et sh sont tres ressemblant et n'arrivent pas a se dissocier sur les axes factoriels
#	- Il serait interessant de recreer des modeles en utilisant les premiers et troisieme axes factoriels qui séparent bien nos variables (sauf aa et sh)


# ------------------------------------------------ FDA -------------------------------------------------
print("FDA")

phoneme.fda.lda<-lda(phoneme.train.label~. ,data=phoneme.train.data)
U <- phoneme.fda.lda$scaling
X <- as.matrix(phoneme.train.data)
Z <- X%*%U

phoneme.fda.lda.test<-lda(phoneme.test.label~. ,data=phoneme.test.data)
Utest <- phoneme.fda.lda.test$scaling
Xtest <- as.matrix(phoneme.test.data)
Ztest <- Xtest%*%Utest

cp1 <- 1
cp2 <- 2

plot(Z[phoneme.train.label=="aa",cp1],Z[phoneme.train.label=="aa",cp2],xlim=range(Z[,1]),ylim=range(Z[,2]),xlab="Z1",ylab="Z2")
points(Z[phoneme.train.label=="ao",cp1],Z[phoneme.train.label=="ao",cp2],pch=2,col="blue")
points(Z[phoneme.train.label=="dcl",cp1],Z[phoneme.train.label=="dcl",cp2],pch=3,col="red")
points(Z[phoneme.train.label=="iy",cp1],Z[phoneme.train.label=="iy",cp2],pch=4,col="pink")
points(Z[phoneme.train.label=="sh",cp1],Z[phoneme.train.label=="sh",cp2],pch=5,col="yellow")
legend("topleft", inset=.05, title="Phoneme", c("aa", "ao", "dcl","iy","sh"), fill=c("black","blue","red","pink","yellow"), horiz=TRUE)

#  		--- LDA - 5.47% d erreur ---
phoneme.fda.lda <- lda(phoneme.train.label~.,data=as.data.frame(Z))
phoneme.fda.lda.pred <- predict(phoneme.fda.lda, newdata=as.data.frame(Ztest))
phoneme.fda.lda.perf <- table(phoneme.test.label,phoneme.fda.lda.pred$class)
phoneme.fda.lda.error <- 1 - sum(diag(phoneme.fda.lda.perf))/(nrow(phoneme.test))
print("LDA avec FDA : ")
print(phoneme.fda.lda.perf)
print(phoneme.fda.lda.error)

#		--- QDA - 5.67% d erreur ---
phoneme.fda.qda <- qda(phoneme.train.label~.,data=as.data.frame(Z))
phoneme.fda.qda.pred <- predict(phoneme.fda.qda, newdata=as.data.frame(Ztest))
phoneme.fda.qda.perf <- table(phoneme.test.label,phoneme.fda.qda.pred$class)
phoneme.fda.qda.error <- 1 - sum(diag(phoneme.fda.qda.perf))/(nrow(phoneme.test))
print("QDA avec FDA : ")
print(phoneme.fda.qda.perf)
print(phoneme.fda.qda.error)

#		--- Regression logistique - 5.27% d'erreur ---
phoneme.fda.glmnet <- glmnet(as.matrix(Z),y=phoneme.train.label,family="multinomial")
phoneme.fda.glmnet.pred <- predict(phoneme.fda.glmnet,newx=as.matrix(Ztest),type="response",s=phoneme.fda.glmnet$lambda.min)
# phoneme.glmnet.pred est un tableau 3 dimensions :
#	- La premiere est sur nos obsvervations (1500 dans l'ensemble de test)
#	- La deuxieme est sur nos types de phonemes (5types de phonemes)
#	- La troisieme est sur l'iteration
phoneme.fda.glmnet.res<-c(rep(0,1500))
for (i in 1:1500)
{
	class <- ""
	res<-which.max(phoneme.fda.glmnet.pred[i,1:5,100])
	{
		if(res==1)
		{
			class <- "aa"
		}
		else if(res==2){
			class <- "ao"
		}
		else if(res==3){
			class <- "dcl"
		}
		else if(res==4){
			class <- "iy"
		}
		else{
			class <- "sh"
		}
	}
	phoneme.fda.glmnet.res[i] <- class 
}
phoneme.fda.glmnet.perf <- table(phoneme.test.label,phoneme.fda.glmnet.res)
phoneme.fda.glmnet.error <- 1 - sum(diag(phoneme.fda.glmnet.perf))/(nrow(phoneme.test))
print("Regression logistique avec FDA : ")
print(phoneme.fda.glmnet.perf)
print(phoneme.fda.glmnet.error)

#		--- KNN - 5% d'erreur - koptimal 7 ---
phoneme.fda.knn.error<-rep(0,20)
for(k in 1:10)
{
	phoneme.fda.knn <- knn(as.data.frame(Z), as.data.frame(Ztest), phoneme.train.label,k=k)
	phoneme.fda.knn.error[k] <- (length(which(FALSE==(phoneme.fda.knn==phoneme.test.label))))/length(phoneme.test.label)

}
print("KNN avec FDA : ")
print(phoneme.fda.knn.error)


#		--- Classifiation tree - 5.2% d'erreur ---
print("TREE - Errors : ")
phoneme.fda.tree<- tree(phoneme.train.label~ ., data=as.data.frame(Z)) 
phoneme.fda.tree.pred<-predict(phoneme.fda.tree, as.data.frame(Ztest), type="class")
phoneme.fda.tree.perf <- table(phoneme.fda.tree.pred, phoneme.test.label)
phoneme.fda.tree.error <- (sum(phoneme.fda.tree.perf)-sum(diag(phoneme.fda.tree.perf)))/nrow(phoneme.test.data)
print(phoneme.fda.tree.perf)
print(phoneme.fda.tree.error)


#		--- Classifieur bayesien naif - 5.6% d'erreur ---
print("BAYES - Errors : ")
phoneme.fda.naive<- naiveBayes(phoneme.train.label~., data=as.data.frame(Z))
phoneme.fda.naive.pred<-predict(phoneme.fda.naive,newdata=as.data.frame(Ztest))
phoneme.fda.naive.perf <-table(phoneme.test.label,phoneme.fda.naive.pred)
phoneme.fda.naive.error <- 1-sum(diag(phoneme.fda.naive.perf))/nrow(phoneme.test.data)
print(phoneme.fda.naive.perf)
print(phoneme.fda.naive.error)

# ------------------------------------------------ FDA + ACP -------------------------------------------------

print("FDA + ACP")
phoneme.fda.lda<-lda(phoneme.train.label~. ,data=as.data.frame(phoneme.acp.train.scores[,1:5]))
U <- phoneme.fda.lda$scaling
X <- as.matrix(phoneme.acp.train.scores[,1:5])
Z <- X%*%U

phoneme.fda.lda.test<-lda(phoneme.test.label~. ,data=phoneme.acp.test.scores[,1:5])
Utest <- phoneme.fda.lda.test$scaling
Xtest <- as.matrix(phoneme.acp.test.scores[,1:5])
Ztest <- Xtest%*%Utest

cp1 <- 1
cp2 <- 2

plot(Z[phoneme.train.label=="aa",cp1],Z[phoneme.train.label=="aa",cp2],xlim=range(Z[,1]),ylim=range(Z[,2]),xlab="Z1",ylab="Z2")
points(Z[phoneme.train.label=="ao",cp1],Z[phoneme.train.label=="ao",cp2],pch=2,col="blue")
points(Z[phoneme.train.label=="dcl",cp1],Z[phoneme.train.label=="dcl",cp2],pch=3,col="red")
points(Z[phoneme.train.label=="iy",cp1],Z[phoneme.train.label=="iy",cp2],pch=4,col="pink")
points(Z[phoneme.train.label=="sh",cp1],Z[phoneme.train.label=="sh",cp2],pch=5,col="yellow")
legend("topleft", inset=.05, title="Phoneme", c("aa", "ao", "dcl","iy","sh"), fill=c("black","blue","red","pink","yellow"), horiz=TRUE)

#  		--- LDA - 11% d erreur ---
phoneme.fda.lda <- lda(phoneme.train.label~.,data=as.data.frame(Z))
phoneme.fda.lda.pred <- predict(phoneme.fda.lda, newdata=as.data.frame(Ztest))
phoneme.fda.lda.perf <- table(phoneme.test.label,phoneme.fda.lda.pred$class)
phoneme.fda.lda.error <- 1 - sum(diag(phoneme.fda.lda.perf))/(nrow(phoneme.test))
print("LDA avec FDA + ACP : ")
print(phoneme.fda.lda.error)

#		--- QDA - 10.73% d erreur ---
phoneme.fda.qda <- qda(phoneme.train.label~.,data=as.data.frame(Z))
phoneme.fda.qda.pred <- predict(phoneme.fda.qda, newdata=as.data.frame(Ztest))
phoneme.fda.qda.perf <- table(phoneme.test.label,phoneme.fda.qda.pred$class)
phoneme.fda.qda.error <- 1 - sum(diag(phoneme.fda.qda.perf))/(nrow(phoneme.test))
print("QDA avec FDA + ACP : ")
print(phoneme.fda.qda.error)

#		--- Regression logistique - 11.4% d'erreur ---
phoneme.fda.glmnet <- glmnet(as.matrix(Z),y=phoneme.train.label,family="multinomial")
phoneme.fda.glmnet.pred <- predict(phoneme.fda.glmnet,newx=as.matrix(Ztest),type="response",s=phoneme.fda.glmnet$lambda.min)
# phoneme.glmnet.pred est un tableau 3 dimensions :
#	- La premiere est sur nos obsvervations (1500 dans l'ensemble de test)
#	- La deuxieme est sur nos types de phonemes (5types de phonemes)
#	- La troisieme est sur l'iteration
phoneme.fda.glmnet.res<-c(rep(0,1500))
for (i in 1:1500)
{
	class <- ""
	res<-which.max(phoneme.fda.glmnet.pred[i,1:5,100])
	{
		if(res==1)
		{
			class <- "aa"
		}
		else if(res==2){
			class <- "ao"
		}
		else if(res==3){
			class <- "dcl"
		}
		else if(res==4){
			class <- "iy"
		}
		else{
			class <- "sh"
		}
	}
	phoneme.fda.glmnet.res[i] <- class 
}
phoneme.fda.glmnet.perf <- table(phoneme.test.label,phoneme.fda.glmnet.res)
phoneme.fda.glmnet.error <- 1 - sum(diag(phoneme.fda.glmnet.perf))/(nrow(phoneme.test))
print("Regression logistique avec FDA + ACP : ")
print(phoneme.fda.glmnet.error)

#		--- KPPV - 10.47% d'erreur - koptimal 16 ---
phoneme.fda.knn.error<-rep(0,20)
for(k in 1:20)
{
	phoneme.fda.knn <- knn(as.data.frame(Z), as.data.frame(Ztest), phoneme.train.label,k=k)
	phoneme.fda.knn.error[k] <- (length(which(FALSE==(phoneme.fda.knn==phoneme.test.label))))/length(phoneme.test.label)

}
print("KNN avec FDA + ACP : ")
print(phoneme.fda.knn.error)

#		--- Classifiation tree - 15.27% d'erreur ---
print("TREE - Errors avec FDA + ACP : ")
phoneme.fda.tree<- tree(phoneme.train.label~ ., data=as.data.frame(Z)) 
phoneme.fda.tree.pred<-predict(phoneme.fda.tree, as.data.frame(Ztest), type="class")
phoneme.fda.tree.perf <- table(phoneme.fda.tree.pred, phoneme.test.label)
phoneme.fda.tree.error <- (sum(phoneme.fda.tree.perf)-sum(diag(phoneme.fda.tree.perf)))/nrow(phoneme.test.data)
print(phoneme.fda.tree.error)


#		--- Classifieur bayesien naif - 11.2% d'erreur ---
print("BAYES - Errors avec FDA + ACP : ")
phoneme.fda.naive<- naiveBayes(phoneme.train.label~., data=as.data.frame(Z))
phoneme.fda.naive.pred<-predict(phoneme.fda.naive,newdata=as.data.frame(Ztest))
phoneme.fda.naive.perf <-table(phoneme.test.label,phoneme.fda.naive.pred)
phoneme.fda.naive.error <- 1-sum(diag(phoneme.fda.naive.perf))/nrow(phoneme.test.data)
print(phoneme.fda.naive.error)

# L AFD seule nous donne des meilleurs résultats car une meilleure séparation des clusters


# ------------------------------------------------ INTERPRETATION -------------------------------------------------

# Les phonemes aa et ao sont tres ressemblants, la plupart des errerus de classification concernent ces deux phonemes.
#Pour le taux d'erreur je choisis le taux de mauvaise classification

