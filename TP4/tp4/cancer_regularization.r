library(glmnet)

cancer = read.table("data/r_breast_cancer.data.txt",header=T,sep=",")

# ====
# Separation des donnees cancer en data et label 
# ====
data <- cancer
data$Time <- NULL
label <- cancer$Time

data.dim <- dim(data)

# 1 - Separation des donnees en test, app
napp <- round(2/3*data.dim[1])
indice <- sample(1:data.dim[1], napp)

data.train <- data[indice,]
data.test <- data[-indice,]

data.train.dim <- dim(data.train)

label.train <- label[indice]
label.test <- label[-indice]

# 2 - Seperation des data de train en train et cv 
ncv <- round(1/3*napp)
indice.cv <- sample(1:data.train.dim[1],ncv)

data.cv <- data.train[indice.cv,]
data.train <- data.train[-indice.cv,]

# 2 - Centre reduire en fonction de notre apprentissage
# Scaling donnees app
data.train.sd <- apply(data.train, 2, sd)
data.train.mean <- colMeans(data.train)
data.train <- as.data.frame(scale(data.train))

label.cv <- label.train[indice.cv]
label.train <- label.train[-indice.cv]

# Scaling donnees cv
for(i in 1:data.dim[2]){
  data.cv[,i]<-(data.cv[,i]-data.train.mean[i])/data.train.sd[i]
}

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

data.cv <- model.matrix(label.cv~.,data.cv)

fit <- glmnet(data.train, label.train, lambda = cv.out$lambda.min, alpha=0)
ridge.pred <- predict(fit, s=cv.out$lambda.min,newx=data.cv)
print(mean(label.cv-ridge.pred)^2)


# ==== Lasso Regression ===
cv.lasso.out <- cv.glmnet(data.train, label.train, alpha = 1)
plot(cv.lasso.out)

fit.lasso <- glmnet(data.train, label.train, lambda = cv.lasso.out$lambda.min, alpha=0)
ridge.pred <- predict(fit.lasso, s=cv.out$lambda.min,newx=data.cv)
print(mean(label.cv-ridge.pred)^2)

plot(cv.lasso.out$lambda, cv.lasso.out$cvm)

# comme dans le cours, ce serait aussi simpa de pouvoir afficher l'evolution du R^2 en fonction du lambda
# on va donc devoir le calculer
label.train.mean = mean(label.train)
tss = sum((label.cv-label.train.mean)^2)
lasso.r2 <- rep(1,length(cv.lasso.out$lambda))
diff <- cv.lasso.out$cvm / tss
lasso.r2  <- lasso.r2 - diff

plot(cv.lasso.out$lambda, lasso.r2)

# == CHOSES A FAIRE ==
# Utiliser aussi le package rda, histoire de tester la derniere solution possible

# === Question PROF ===
# Dans nos jeux de donnees on a des echelles tres differentes pour nos parametres
# doit-on du coup scale nos features ?
# Si oui, dans ce cas la comment le faire pour la regression de ridge, car
# dans l'exemple du cours, on a rajouter l'intercept au debut 
# or on ne doit pas scale l'intercept si ?

# Est ce que le faire de cette facon pour le model.matrix revient bien au meme 
# Pck j'ai l'impression que oui, et que ca ne rajoute juste qu'un intercept ..
# mais je prefererai en etre sur

# Avec la fonction cv.glmnet on fait deja au final notre CV
# du coup je ne sais pas trop trop si je dois utiliser directement les 
# datas de test ou si c'est mieux de faire avec les data.cvs
# pck au final plus on a de datas pour notre modele, mieux c'est !