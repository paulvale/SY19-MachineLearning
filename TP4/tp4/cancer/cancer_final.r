# CLEAN WORKSPACE
rm(list = ls())

# LOAD DATA
library(corrplot)
library(leaps)
source("cancer/cancer_dataSeparation.r")

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
plot(reg.fit, scale="r2")
#readline(prompt="Press [enter] to continue")
Formula <- getFormulas(data.colnames, reg.order, "label.cv")
formula.elastic <- getFormulas(data.colnames, reg.order, "label.train")

# On prend generalement 5 ou 10 comme valeur de K 
# C'est deux valeurs correspondant a un bon compromis biais-vairance
# on va donc ici essayer avec les 2 differentes valeurs et observer nos resultats

# Initialisation de nos vecteurs erreurs pour nos 32 modeles differents
# On va choisir 4 techniques differentes : linear, lasso, ridge et elasticNet
valueOfLambda <- seq(0,1,by=0.1)

error.linear <- rep(0,data.train.dim[2])
error.lasso <- rep(0,data.train.dim[2])
error.ridge <- rep(0,data.train.dim[2])
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
  #readline(prompt="Press [enter] to continue")
}
# VISUALISATION
num_pred <- c(1:length(reg.order))

plot(num_pred,error.linear, type="l")
print(num_pred[which.min(error.linear)])
print(min(error.linear))

