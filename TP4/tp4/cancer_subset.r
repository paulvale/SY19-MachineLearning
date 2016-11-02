# LOAD DATA
library(corrplot)
library(leaps)
source("cancer_dataSeparation.r")

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

# ====
# Utilisation de la regression lineaire avec le modele de base
# ====
# 1 - Recuperation des datas
data.train <- getDataTrainScale()
data.train.dim <- dim(data.train)

label.train <- getLabelTrain()

# 2 - Separation des data de train en train et cv 
ncv <- round(1/3*data.train.dim[1])
indice.cv <- sample(1:data.train.dim[1],ncv)

data.cv <- data.train[indice.cv,]
data.train <- data.train[-indice.cv,]

label.cv <- label.train[indice.cv]
label.train <- label.train[-indice.cv]
  
# Forward stepwise selection
reg.fit <- regsubsets(label.train~. , data=data.train, method='forward', nvmax=33)
reg.order <- reg.fit$vorder
reg.order <- reg.order - 1
reg.order <- reg.order[2:length(reg.order)]
data.colnames <- colnames(data.train)
plot(reg.fit, scale="r2")

Formula <- getFormulas(data.colnames, reg.order, "label.train")
num_pred <- c(1:length(reg.order))
err <- c(1:length(reg.order))

for(i in 1:(length(reg.order))){
  reg <- lm(Formula[i], data=data.train)
  pred <- predict(reg,newdata=data.cv)
  err[i]<-mean((label.cv-pred)^2)
}

plot(num_pred,err, type="l")

# === Observation ===
# Comme on a pu le voir dans le cours, 
# au final, on a ici une variance qui est tout de meme assez importante sur nos courbes,
# cependant, on arrive tout de meme deja a en degager une tendance
# en effet, il se trouve que le nombre de predicteur optimum se situe entre 0 et 10
# voir 15 mais jamais apres 15 ...
# Apres avoir depasser ce nombre de predicteurs, l'erreur est alors souvent beaucoup 
# trop eleve

# === Avancement ===
# Il serait donc maintenant interessant de mettre ce test en place avec 
# une K-fold cross validation afin d'avoir une variance de nos courbres d'autant moins 
# importante


# === K-Fold CV ===
# Definitions de nos parametres souhaites 
# Ici, on ne veut plus avoir un ensemble cv, test et app
# car le but de K-fold cv, c'est de tester toutes les differentes
# possibilites dans les datas de app pour gagner en precision
# et donc perdre en variabilite dans nos estimations

# On reprend donc notre jeu de datas de depart et on le divise juste en 2
# On a garde en memoire les indices de nos jeux de donnees, afin 
# d'utiliser exactement les memes ( napp, indice)
data.train.cv <- getDataTrainScale()
data.train.dim.cv <- dim(data.train.cv)
label.train.cv <- getLabelTrain()

formula.cv <- getFormulas(data.colnames, reg.order,"label.function.cv")

# On prend generalement 5 ou 10 comme valeur de K 
# C'est deux valeurs correspondant a un bon compromis biais-vairance
# on va donc ici essayer avec les 2 differentes valeurs et observer nos resultats

folds = sample(1:10,data.train.dim.cv[1], replace=TRUE)
CV <- rep(0,data.train.dim.cv[2])
for(i in (1:data.train.dim.cv[2])){
  for(k in (1:10)){
    label.function.cv <- label.train.cv[folds!=k]
    reg <- lm(formula = formula.cv[i],data=data.train.cv[folds!=k,])
    pred <- predict(reg, newdata=data.train.cv[folds==k,])
    CV[i] <- CV[i] + sum((label.train.cv[folds==k]-pred)^2)
  }
  CV[i]<-CV[i]/data.train.dim.cv[1]
}

plot(num_pred,CV, type="l")

# === Observation ===
# Comme on pouvait l'esperer, notre variance entre nos courbes a bien diminue
# Bien qu'au debut nous pouvions penser que prendre entre 0 et 5 predicteurs suffise
# on s'apercoit dorenavant qu'il est d'autant plus interessant de rester entre 5 et 10 
# predicteurs pour avoir l'erreur la plus faible possible
# Cependant, on n'oublie pas de regarder l'echelle et on se rend compte que l'erreur
# est tout de meme eleve ... avec 1000 au minimum ..

# Il serait donc interessant de voir si la regularisation ne serait pas plus interessant
# pour notre modele que le subset selection


# ===
# Derniere chose a faire :
# ===
# Il faudrait pour finir choisir un nombre de parametre et les parametres en question
# puis lancer une regression en fonction de tout cela sur notre ensemble de test
# afin d'avoir une valeur finale de notre erreur et donc 
# pouvoir evaluer notre modele en fonction des prochains modeles que nous allons construire


