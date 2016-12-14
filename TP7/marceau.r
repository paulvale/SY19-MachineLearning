i<- 0
response <- rep(0,6)
for(i in 0:216){
	response[y[i]] <- response[y[i]]+1
}

data <- X
data.dim <- dim(X)

# 1 - Separation des donnees en test, app
napp <- round(0.8*data.dim[1])
indice <- sample(1:data.dim[1], napp)
train.data <- data[indice,1:4200]
test.data <- data[-indice,1:4200]
train.label<- y[indice]
test.label <- y[-indice]

i<- 0
response <- rep(0,6)
for(i in 1:173){
  response[train.label[i]] <- response[y[i]]+1
}

