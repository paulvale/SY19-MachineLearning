# LOAD DATA
install.packages("FNN")
install.packages("hydroGOF")
library(MASS)
library(pROC)
library(nnet)
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
phoneme.train.data <- phoneme.train[,1:257]
phoneme.train.label <- phoneme.train$g
phoneme.test <- rbind(aa.test, sh.test, ao.test, dcl.test, iy.test)
phoneme.test.data <- phoneme.test[,1:257]
phoneme.test.label <- phoneme.test[,258]


# ------------------------------------------------ CONSTRUCTION OF MODELS -------------------------------------------------

#on doit ici predire une classe et non une valeur, on utilisera donc des methodes de classification et non de regression

#  		--- LDA - 8.6% d erreur ---
phoneme.lda <- lda(phoneme.train.label~.,data=phoneme.train.data)
phoneme.lda.pred <- predict(phoneme.lda, newdata=phoneme.test.data)
phoneme.lda.perf <- table(phoneme.test.label,phoneme.lda.pred$class)
phoneme.lda.error <- 1 - sum(diag(phoneme.lda.perf))/(nrow(phoneme.test))

#		--- QDA - 18.93% d erreur ---
phoneme.qda <- qda(phoneme.train.label~.,data=phoneme.train.data)
phoneme.qda.pred <- predict(phoneme.qda, newdata=phoneme.test.data)
phoneme.qda.perf <- table(phoneme.test.label,phoneme.qda.pred$class)
phoneme.qda.error <- 1 - sum(diag(phoneme.qda.perf))/(nrow(phoneme.test))


# pourquoi pas faire une regression logistique en predisant les probabilité de la classe 1, 2, 3 ,4 et 5
# meme chose nen regression lineaire
# meme chose avec des kppv

#		--- Regression lineaire ---
#phoneme.glm <- glm(phoneme.train.label~.,data=phoneme.train.data, family=binomial)
#phoneme.glm.pred <- predict(phoneme.glm,newdata=phoneme.test.data,type='response') #donne les prob predites
#phoneme.glm.perf <- table(phoneme.test.label,phoneme.glm.pred>0.2)
#phoneme.glm.err <- 1 - sum(diag(phoneme.glm.perf))/(nrow(phoneme.test))

#ne marche pas car plusieurs classes (plus que 2)

#phoneme.glm.aa <- f1(phoneme.train.label,'aa')
#phoneme.glm.ao <- f1(phoneme.train.label,'ao')
#phoneme.glm.dcl <- f1(phoneme.train.label,'dcl')
#phoneme.glm.iy <- f1(phoneme.train.label,'iy')
#phoneme.glm.sh <- f1(phoneme.train.label,'sh')
#phoneme.glm.global <- matrix(c(phoneme.glm.aa, phoneme.glm.ao,phoneme.glm.dcl,phoneme.glm.iy,phoneme.glm.sh),ncol=5,byrow = F)
#model1 <- multinom(phoneme.glm.global~.,data=phoneme.train.data)

#ne marche pas non plus

#		--- KPPV - 45% d'erreur---
phoneme.kppv.error <- rep(0,30)
for(k in 2:8)
{
    phoneme.kppv.pred <- kppv(phoneme.train.data, phoneme.train.label,k,phoneme.test.data)
    phoneme.kppv.perf <- table(phoneme.test.label,phoneme.kppv.pred)
	phoneme.kppv.error[k] <- 1 - sum(diag(phoneme.kppv.perf))/(nrow(phoneme.test))
}
#Les kppv nous donne un taux d'erreur compris entre 45% et 59%, avec un k optimal à 3










