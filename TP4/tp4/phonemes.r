# LOAD DATA

library(MASS)
library(pROC)
library(nnet)
library(glmnet)
library(leaps)
library("FNN", character.only = TRUE)
library("hydroGOF", character.only = TRUE)
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

# Toutes les datas ont plus ou moins une variance et une une moyenne identique, nous centrons et reduisons quand meme 
# pour etre sur de rendre ces variables comparables

phoneme[,2:257] <- as.data.frame(scale(phoneme[,2:257]))


# -------------------------------------------- DATA SEPARATION --------------------------------------------------------

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

#regarder si toutes les varibles sont importants (regarder la correlation des variables)
#regularisation : regarder si rajouter un terme on a pas un meilleur modele (regarder la correlation des variables)
#facteur analysis : changemet de repere 
#faire un ACP pour peut etre reduire


# ------------------------------------------------ SUBSET SELECTION -------------------------------------------------
reg.fit<- regsubsets(phoneme.train.label~.,data=phoneme.train.data,method='forward',nvmax=256)
summary.regsubsets <- summary(reg.fit)
summary.regsubsets.which<-summary.regsubsets$which #permet de savoir quels variables sont dans quels modeles. (il faut decaler de 2)
LDA_ERROR <- matrix(0,ncol=2,nrow=250)
QDA_ERROR <- matrix(0,ncol=2,nrow=250)
KNN_ERROR <- matrix(0,ncol=2,nrow=250)
lda.min <- 100
qda.min <- 100
knn.min <- 100
lda.subset <- summary.regsubsets.which[2,3:257]
qda.subset <- summary.regsubsets.which[2,3:257]
knn.subset <- summary.regsubsets.which[2,3:257]
k.opt <- 0
for(i in 2:250)#ca sert a rien de le faire jusqu'a 256 on a deja les resualtas plus haut.
{
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
	
}
print("Apres subset selection : ")
print("LDA error minimale : ")
print(lda.min)
print("QDA error minimale : ")
print(qda.min)
print("KNN error minimale : ")
print(knn.min)

# On peut deduire de ce modele qu'une selection de variable reduit notre erreur moyenne de nos modele.
# Pour l analyse discriminnate lineaire  l erreur descend a 7.93% lorsqu'on garde seulement 2 variables
# Pour l analyse discriminante quadratique l erreur descend a 7.4% lorsqu on garde 37 variables
# Pour les KNN l erreur descend a 7.67% lorsqu'on garde 41 variables avec un k optimnal qui reste a 8

# ------------------------------------------------ REGRESSION RIDGE & LASSO -------------------------------------------------

# Permet de estimer un modele avec des variables fortement correlees
# Avantage de la regression ridge : les variables explicatrices tres correlees se combinent pour se renforcer mutuellement
# Cette methode garde toutes les variables mais aucun moyen de savoir lesquelles ont plus de poids que d'autres

# La methode LASSO met a 0 les variables peu explicatives
# Si les variables sont correlees, l'algorithme en choisi une et la met a 0
# ------------------------------------------------ ANALYSE EN COMPOSANTE PRINCIPALE -------------------------------------------------

# Nous avons effectué une analyse en composante principale 
# l'idee serait de creer un nouvel axe factoriel et de creer un modele a partir de la

# Lorsqu'on regarde nos vecteurs propres, on remarque qu on peut en garder 2 ou 3 pour expliquer nos 257 variables de maniere efficace

# ------------------------------------------------ INTERPRETATION -------------------------------------------------

# Les phonemes aa et ao sont tres ressemblants, la plupart des errerus de classification concernent ces deux phonemes.


