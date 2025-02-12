# CLEAN WORKSPACE
rm(list = ls())

# LOAD DATA
library(corrplot)
library(leaps)
library(glmnet)
library(pls)
source("cancer/cancer_dataSeparation.r")


# Fonction permettant de recuperer toutes les formules pour 
# les regressions lineaires 
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

# ==============================
# ======== CHOIX DU MODELE =====
# ==============================

# 1 - Recuperation des datas
data.train <- getDataTrainScale()
data.train.dim <- dim(data.train)

label.train <- getLabelTrain()

# 2 - Forward stepwise selection
reg.fit <- regsubsets(label.train~. , data=data.train, method='forward', nvmax=33)
reg.order <- reg.fit$vorder
reg.order <- reg.order - 1
reg.order <- reg.order[2:length(reg.order)]
data.colnames <- colnames(data.train)
plot(reg.fit, scale="r2", cex=0.5)
readline(prompt="Press [enter] to continue")
Formula <- getFormulas(data.colnames, reg.order, "label.cv")

# On prend generalement 5 ou 10 comme valeur de K 
# C'est deux valeurs correspondant a un bon compromis biais-vairance

# Initialisation de nos vecteurs erreurs pour nos 32 modeles differents
# On va choisir 4 techniques differentes : linear, lasso, ridge et elasticNet
valueOfLambda <- seq(0,1,by=0.1)

error.linear <- rep(0,data.train.dim[2])
error.elasticNet <- matrix(0,nrow=data.train.dim[2], ncol=length(valueOfLambda))

# Creation des partitions pour la crossValidation sous linear
folds = sample(1:10,data.train.dim[1], replace=TRUE)
for(i in (1:data.train.dim[2])){
  # 1 - Linear
  for(k in (1:10)){
    label.cv <- label.train[folds!=k]
    reg <- lm(formula = Formula[i],data=data.train[folds!=k,])
    pred <- predict(reg, newdata=data.train[folds==k,])
    error.linear[i] <- error.linear[i] + sum((label.train[folds==k]-pred)^2)
  }
  error.linear[i]<-error.linear[i]/data.train.dim[1]
  
  # 2 - Preparation de la matrix pour les regularisations 
  if( i == 1){
    data.regression <- as.matrix(data.train[,which(reg.order == i),])
  } else {
    data.regression <- cbind(data.regression, data.train[,which(reg.order == i)])
  } 
  
  data.regression <- as.data.frame(data.regression)
  colnames(data.regression)[i] <- i
  data.regression.train <- model.matrix(label.train~.,data.regression)

  # 3 - From Ridge to Lasso
  for(l in 1:length(valueOfLambda)){
    cv.elastic.out <- cv.glmnet(data.regression.train, label.train, alpha = valueOfLambda[l])  
    error.elasticNet[i,l]<-min(cv.elastic.out$cvm)
  }
}

# Dernier modele : PCR (Principal Component regression)
model.pcr <- pcr(label.train~.,data=data.train, validation="CV")
validMSEP = MSEP(model.pcr)


# VISUALISATION
num_pred <- c(1:length(reg.order))

# Linear Regression
plot(num_pred,error.linear, type="l")
print(num_pred[which.min(error.linear)])
print(min(error.linear))
readline(prompt="Press [enter] to continue")

# Ridge to Lasso
for(lambda in 1:length(valueOfLambda)){
  plot(num_pred,error.elasticNet[,lambda], type="l", ylab="MSE")
  print(num_pred[which.min(error.elasticNet[,lambda])])
  print(min(error.elasticNet[,lambda]))
  readline(prompt="Press [enter] to continue")
}

# PCR
validationplot(model.pcr, val.type = "MSEP", legendpos = "topright")


# ====================================
# ======== TEST DU MODELE FINALE =====
# ====================================

# 1 - Reprend les datas
# Train
data.train <- getDataTrainScale()
data.train.dim <- dim(data.train)

label.train <- getLabelTrain()

# Test
data.test <- getDataTestScale()
label.test <- getLabelTest()


# 2 - Calcule du modele et de ses performances
print(min(error.linear))
print(min(error.elasticNet))
print(min(validMSEP$val["CV",,]))

if( (min(error.elasticNet) < min(error.linear)) && (min(error.elasticNet) < min(validMSEP$val["adjCV",,]))){
  print("elastic")
  indexes = which(error.elasticNet == min(error.elasticNet), arr.ind = TRUE)
  numberOfParameters = indexes[1]
  alpha = valueOfLambda[indexes[2]]
  print(numberOfParameters)
  print(alpha)
  
  # Fabrication de la matrix train et test
  data.regression <- as.matrix(data.train[,which(reg.order == 1),])
  data.regression.test <- as.matrix(data.test[,which(reg.order == 1),])
  if( numberOfParameters != 1){
    for( number in 2 : numberOfParameters) {
      data.regression <- cbind(data.regression, data.train[,which(reg.order == number)])
      data.regression <- as.data.frame(data.regression)
      colnames(data.regression)[number] <- number
      
      data.regression.test <- cbind(data.regression.test, data.test[,which(reg.order == number)])
      data.regression.test <- as.data.frame(data.regression.test)
      colnames(data.regression.test)[number] <- number
    }
  }
  
  data.regression <- as.data.frame(data.regression)
  data.regression.train <- model.matrix(label.train~.,data.regression)
  
  data.regression.test <- as.data.frame(data.regression.test)
  data.test <- model.matrix(label.test~.,data.regression.test)
  
  # Calcul du lambda.min
  cv.out <- cv.glmnet(data.regression.train, label.train, alpha = alpha)
  
  # Calcul du modele
  fit <- glmnet(data.regression.train, label.train, lambda=cv.out$lambda.min, alpha = alpha)
  pred <- predict(fit, s=cv.out$lambda.min, newx=data.test)
  
  # Calcul de l'erreur
  error <- mean((label.test-pred)^2)
} else if ((min(error.linear) < min(error.elasticNet)) && (min(error.linear) < min(validMSEP$val["CV",,]))) {
  print("linear")
  print(which.min(error.linear))
  numberOfParameters <- which.min(error.linear)
  FormulaTest <- getFormulas(data.colnames, reg.order, "label.train")
  
  # Calcul du modele
  reg <- lm(formula = FormulaTest[numberOfParameters],data=data.train)
  pred <- predict(reg, newdata=data.test)
  
  # Calcul de l'erreur
  error <- mean((label.test-pred)^2)
} else {
  
  print("pcr")
  print(which.min(validMSEP$val["CV",,]))
  
  # Calcul du modele
  model.pcr <- pcr(label.train~., data=data.train, ncomp=which.min(validMSEP$val["CV",,]))
  summary(fit.pcr)
  pred.pcr <- predict(model.pcr, newdata=data.test)
  
  # Calcul de l'erreur
  print(mean((pred.pcr - label.test)^2))
}

# Affichage de l'erreur
print(error)



