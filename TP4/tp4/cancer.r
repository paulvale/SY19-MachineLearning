# LOAD DATA
library(corrplot)

cancer = read.table("data/r_breast_cancer.data.txt",header=T,sep=",")

linearRegression <- function(parameters, label){
  reg <- lm(label ~., data=parameters)
  print(summary(reg))
  
  # Regression diagnostic
  #plot(fitted(reg))
  #readline(prompt="Press [enter] to continue")
  #plot(resid(reg))
  #readline(prompt="Press [enter] to continue")
  #plot(rstandard(reg))
  #readline(prompt="Press [enter] to continue")
  #plot(rstudent(reg))
  #readline(prompt="Press [enter] to continue")
  #plot(cooks.distance(reg))
  #readline(prompt="Press [enter] to continue")
  #plot(hatvalues(reg))
  #readline(prompt="Press [enter] to continue")
  plot(reg)
  
  plot(label,label,col="black")     
  abline(a=0,b=1)     
  points(label,reg$fitted.values,pch=19,col="red",cex=0.7)
  return(reg)
}

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
data.reg <- linearRegression(data.train, label.train)

# Ici on peut voir que l'on a un R-squared ajuste qui est a ... 0.1126 
# Or , In general, the higher the R-squared, the better the model fits your data.
# Du coup, ce modele ne va pas du tout !!!!
# D'ailleurs on le voit au niveau de notre graphiques ! Les valeurs estimees sont
# trop importantes au debut et pas assez elevees ensuite ...

# on va donc passer a la selection du modele et des parametres afin d'ameliorer
# notre modele.


