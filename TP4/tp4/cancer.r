# LOAD DATA
library(corrplot)
cancer = read.table("data/r_breast_cancer.data.txt",header=T,sep=",")

# ====
# Separation des donnees cancer en data et label 
# ====
data <- cancer
data$Time <- NULL
label <- cancer$Time

data.dim <- dim(data)

# ===
# Donnees de base sur les datas a observer
# ===
min <- apply(data,2,min)
max <- apply(data,2,max)
mean <- colMeans(data)
data.obs <- data.frame(min,mean,max)

# Apres observation de nos donnees, on se rend compte qu'il y a une grande variance au niveau de l'echelle
# de nos predicteurs
# ex : mean area_worst            = 1400
#      mean fractal_dimension_se  = 0.004
# On va donc devoir centre reduire nos variables dans un premier temps 

data.cor <- cor(data)
corrplot(data.cor, type="lower")

# on peut voir aussi qu'il y a des correlations importantes entre certains
# de nos predicteurs 
# On pouvait s'en douter, en effet le worst perimeter va influer sur le mean perimeter
# ...

# ====
# Utilisation de la regression lineaire avec le modele de base
# ====

# 1 - Separation des donnees en test et app
napp <- round(2/3*data.dim[1])
indice <- sample(1:data.dim[1], napp)

data.train <- data[indice,]
data.test <- data[-indice,]

label.train <- label[indice]
label.test <- label[-indice]

# 2 - Centre reduire en fonction de notre apprentissage
# Scaling donnees app
data.train.sd <- apply(data.train, 2, sd)
data.train.mean <- colMeans(data.train)
data.train <- as.data.frame(scale(data.train))

# Scaling donnees test
for(i in 1:data.dim[2]){
  data.test[,i]<-(data.test[,i]-data.train.mean[i])/data.train.sd[i]
}

# Linear Regression
data.reg <- lm(label.train ~., data=data.train)
print(summary(data.reg))

# Regression diagnostic
plot(fitted(data.reg))
readline(prompt="Press [enter] to continue")
plot(resid(data.reg))
readline(prompt="Press [enter] to continue")
plot(rstandard(data.reg))
readline(prompt="Press [enter] to continue")
plot(rstudent(data.reg))
readline(prompt="Press [enter] to continue")
plot(cooks.distance(data.reg))
readline(prompt="Press [enter] to continue")
plot(hatvalues(data.reg))



