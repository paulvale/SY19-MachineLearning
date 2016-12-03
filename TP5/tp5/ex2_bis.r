# LOAD DATA
library("leaps")
library("gam")
prostate.data = read.table("data/prostate.data.txt")

data = prostate.data
data$train = NULL
data$gleason = NULL
data$age = NULL
data$pgg43 = NULL
data$lcp = NULL

polyVector <- c(3:15)
data.dim <- dim(data)
errors.matrix <- matrix(0, nrow=length(polyVector), ncol=5)
# On va donc avoir une matrice du type
#          model1  model2 model3 ...
# df = 3
# df = 4
# df =...
folds = sample(1:5,data.dim[1], replace=TRUE)
for(i in polyVector){
  # bs1.1
  for(k in (1:5)){
    reg <- gam(lpsa~bs(lcavol, df=i)+lweight+lbph+svi, data= data[folds!=k,])
    pred <- predict(reg, newdata=data[folds==k,], interval="c")
    errors.matrix[i-2,1]<- errors.matrix[i-2,1] + sum((data$lpsa[folds==k] - pred)^2)
  }
  errors.matrix[i-2,1]<-errors.matrix[i-2,1]/data.dim[1]
  
  # ns1.1
  for(k in (1:5)){
    reg <- gam(lpsa~ns(lcavol, df=i)+lweight+lbph+svi, data= data[folds!=k,])
    pred <- predict(reg, newdata=data[folds==k,], interval="c")
    errors.matrix[i-2,2]<- errors.matrix[i-2,2] + sum((data$lpsa[folds==k] - pred)^2)
  }
  errors.matrix[i-2,2]<-errors.matrix[i-2,2]/data.dim[1]
  
  # ns2.1
  for(k in (1:5)){
    reg <- gam(lpsa~ns(lcavol, df=i)+ns(lweight, df=i)+lbph+svi, data= data[folds!=k,])
    pred <- predict(reg, newdata=data[folds==k,], interval="c")
    errors.matrix[i-2,3]<- errors.matrix[i-2,3] + sum((data$lpsa[folds==k] - pred)^2)
  }
  errors.matrix[i-2,3]<-errors.matrix[i-2,3]/data.dim[1]
  
  # ns3
  for(k in (1:5)){
    reg <- gam(lpsa~ns(lcavol, df=i)+ns(lweight, df=i)+ns(lbph, df=i)+svi, data= data[folds!=k,])
    pred <- predict(reg, newdata=data[folds==k,], interval="c")
    errors.matrix[i-2,4]<- errors.matrix[i-2,4] + sum((data$lpsa[folds==k] - pred)^2)
  }
  errors.matrix[i-2,4]<-errors.matrix[i-2,4]/data.dim[1]
  
  # ss1.1
  for(k in (1:5)){
    reg <- gam(lpsa~s(lcavol, df=i)+lweight+lbph+svi, data= data[folds!=k,])
    pred <- predict(reg, newdata=data[folds==k,], interval="c")
    errors.matrix[i-2,5]<- errors.matrix[i-2,5] + sum((data$lpsa[folds==k] - pred)^2)
  }
  errors.matrix[i-2,5]<-errors.matrix[i-2,5]/data.dim[1]
}


# Ok pour faire une CV pour trouver le df ... 
# Mais a chaque fois on ne trouve le df que de 1 seul parametre
# donc lorsque l'on souhaite avoir par exemple ns3
# qui va utiliser les smoothing splines sur les 3 parametres
# il faudra faire 3CV imbriques 
# dans le sens ou il faut trouver la meilleure combinaison des 3
# dans tous les cas, comment pouvoir aussi reduire un peu notre champ de recherche
# car ici on a
# - 3 variables 
# - 3 methodes differentes 
# - 15 df
# => beaucoup trop de modèles possibles si on le fait exhaustivement ...`

# Pareil peut-on melanger nos splines pour notre modele ou non ?
# c'est a dire pouvons nous avoir des valeurs comme :
# gam(lpsa~s(X)+bs(Y))


# Reponses :
# le meilleur model est donc le ns1.1 avec df = 3
# reg <- gam(lpsa~ns(lcavol, df=i)+lweight+lbph+svi, data= data[folds!=k,])