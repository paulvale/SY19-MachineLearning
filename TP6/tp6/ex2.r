# LOAD DATA
library(mclust)
wine <- read.table("data/wine.txt")


summary(wine)
plot(wine, col=c("red", "blue", "green")[wine$V1])
# Comme on peut le voir sur certaines variables, nos differents types sont difficilement decernables
plot(wine$V4,wine$V6,col=c("red", "blue", "green")[wine$V1])


wineMclustVVV <- Mclust(data = wine, modelNames = "VVV")
#plot(wineMclust)
# Question au prof :
# On a 28 modeles different ok mais du coup cela veut donc dire que nous devons tous les tester afin de trouver
# celui qui nous convient le mieux ?
# Ex : EEE, et VVV vu en cours comme example

# => Pour tester tous les modeles on fait alors
# Mais c'est long ... env. 
# En quoi dans le cours on a reussi a voir avec les plots le meilleur modele ...
# Juste en faisant le summary ca suffit pour dire que c'est le meilleur modele .?
wineMclust <- Mclust(data = wine)
print(summary(wineMclust))
# Normalement on a
#class 1 59
#class 2 71
#class 3 48

# Malheureusement si on ne dit pas que l'on a 3 groupes alors le meilleur modele en propose 4 :
#Mclust EVE (ellipsoidal, equal volume and orientation) model with 4 components:
  
#  log.likelihood   n  df       BIC       ICL
#-2595.432 178 203 -6242.766 -6252.968

#Clustering table:
#  1  2  3  4 
#59 33 21 65 

# On va donc reessayer mais avec cette fois ci que 3 groupes 
wineMclust2 <- Mclust(data = wine, G=3)
print(summary(wineMclust2))

#  Mclust EVE (ellipsoidal, equal volume and orientation) model with 3 components:
  
#  log.likelihood   n  df       BIC      ICL
#-2953.06 178 175 -6812.933 -6814.28

#Clustering table:
#  1  2  3 
# 59 56 63  

# En regardant la classification faite, on se rend compte qu'au final, notre modele a eu du mal a 
# faire la difference entre le producteur 2 et 3 mais a reussi a retrouver a 100% le producteur 1


