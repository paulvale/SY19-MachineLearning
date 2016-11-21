# LOAD DATA
cancer = read.table("data/r_breast_cancer.data.txt",header=T,sep=",")

# ====
# Separation des donnees cancer en data et label 
# ====
data <- cancer
data$Time <- NULL
label <- cancer$Time

data.dim <- dim(data)

# 1 - Separation des donnees en test, app
napp <- round(8/10*data.dim[1])
indice <- sample(1:data.dim[1], napp)

data.train <- data[indice,]
data.test <- data[-indice,]

data.train.dim <- dim(data.train)

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


getDataTrainScale <- function(){
  return(data.train)
}

getLabelTrain <- function(){
  return(label.train)
}

getDataTestScale <- function(){
  return(data.test)
}

getLabelTest <- function(){
  return(label.test)
}