# LOAD DATA
library(corrplot)
library(leaps)

cancer = read.table("data/r_breast_cancer.data.txt",header=T,sep=",")

getFormulas <- function(col, order) {
  result <- vector(mode="character", length=length(order))
  for(i in 1:length(order)){
    if(i == 1){
      result[i] <- paste(c("label.train ~ ",col[order[i]]),collapse = '')
    } else {
      result[i] <- paste(c(result[i-1], col[order[i]]), collapse = ' + ')
    }
  }
  return(result)
}

# ====
# Separation des donnees cancer en data et label 
# ====
data <- cancer
data$Time <- NULL
label <- cancer$Time

data.dim <- dim(data)

# ====
# Utilisation de la regression lineaire avec le modele de base
# ====

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

# Forward stepwise selection
reg.fit <- regsubsets(label.train~. , data=data.train, method='forward', nvmax=33)
reg.order <- reg.fit$vorder
reg.order <- reg.order - 1
reg.order <- reg.order[2:length(reg.order)]
data.colnames <- colnames(data.train)
plot(reg.fit, scale="r2")

Formula <- getFormulas(data.colnames, reg.order)
num_pred <- c(1:length(reg.order))
err <- c(1:length(reg.order))
for(i in 1:(length(reg.order))){
  reg <- lm(Formula[i], data=data.train)
  pred <- predict(reg,newdata=data.cv)
  err[i]<-mean((label.cv-pred)^2)
}

plot(num_pred,err)