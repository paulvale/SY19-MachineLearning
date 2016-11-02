library(glmnet)
source("cancer_dataSeparation.r")

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
data.train <- model.matrix(label.train ~.,data.train)
cv.out <- cv.glmnet(data.train, label.train, alpha = 0)
plot(cv.out)

# ==== Lasso Regression ===
cv.lasso.out <- cv.glmnet(data.train, label.train, alpha = 1)
plot(cv.lasso.out)

#plot(cv.lasso.out$lambda, cv.lasso.out$cvm)

# comme dans le cours, ce serait aussi simpa de pouvoir afficher l'evolution du R^2 en fonction du lambda
# on va donc devoir le calculer
#label.train.mean = mean(label.train)
#tss = sum((label.cv-label.train.mean)^2)
#lasso.r2 <- rep(1,length(cv.lasso.out$lambda))
#diff <- cv.lasso.out$cvm / tss
#lasso.r2  <- lasso.r2 - diff

#plot(cv.lasso.out$lambda, lasso.r2)

# == CHOSES A FAIRE ==
# Utiliser aussi le package rda, histoire de tester la derniere solution possible