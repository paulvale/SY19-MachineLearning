library(glmnet)
source("cancer/cancer_dataSeparation.r")

# ====
# Utilisation de la regression lineaire avec le modele de base
# ====
# 1 - Recuperation des datas
data.train <- getDataTrainScale()
data.train.dim <- dim(data.train)

label.train <- getLabelTrain()

# ==== Ridge Regression ===
# Pour le ridge regression, on a besoin de notre matrice avec deja un intercept, 
# il faut donc lui ajouter dans nos datas
# ===
# Nous ne pouvions pas le faire avant, car nous avons du centre reduire nos valeurs 
# il faut donc le faire apres
# A noter ici que nous allons utiliser la fonction cv.glmnet qui utilise deja
# le CV 

# Ajout de l'intercept a nos data
# On a besoin d'utiliser cette fonction pour que ca marche avec glmnet
data.regression.train <- model.matrix(label.train ~.,data.train)
cv.out <- cv.glmnet(data.regression.train, label.train, alpha = 0)
plot(cv.out)

# ==== Lasso Regression ===
cv.lasso.out <- cv.glmnet(data.regression.train, label.train, alpha = 1)
plot(cv.lasso.out)

# ==== Elastic Net ===
valueOfLambda <- seq(0,1,by=0.1)
cvmFromLambda <- rep(0, length(valueOfLambda))

for(i in 1:length(valueOfLambda)){
  cv.elastic.out <- cv.glmnet(data.regression.train, label.train, alpha = valueOfLambda[i])  
  cvmFromLambda[i]<-min(cv.elastic.out$cvm)
}

plot(valueOfLambda, cvmFromLambda)
# valueOfLambda[which.min(cvmFromLambda)]
# histoire de prendre le minimum de la value de lambda qui nous interesse
print(valueOfLambda[which.min(cvmFromLambda)])
print(min(cvmFromLambda))
