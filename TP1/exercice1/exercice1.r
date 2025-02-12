
# LOAD DATA
prostate.data = read.table("prostate.data.txt")

# INSTALLATION PACKAGES
install.packages("FNN")
#install.packages("automap")
#install.packages("spacetime")
#install.packages("sp")
#install.packages("zoo")
install.packages("hydroGOF")

# LOAD PACKAGES
library("FNN", character.only = TRUE)
#library("automap", character.only = TRUE)
#library("spacetime", character.only = TRUE)
#library("sp", character.only = TRUE)
#library("zoo", character.only = TRUE)
library("hydroGOF", character.only = TRUE)

# ===================================
# REGRESSION PAR LA METHODE DES KPPVS
# ===================================

plot(prostate.data,col='blue',pch=20)

readline(prompt="Press [enter] to continue")
# Valeur de train sans etiquette + Normalization of their features
train = subset(prostate.data,train == T, select=c(lcavol,lweight,lbph,age))
train = data.frame(lapply(train,function(x) scale(x)))
# Valeur de test sans etiquette + Normalization of their features
test = subset(prostate.data,train == F, select=c(lcavol,lweight,lbph,age))
test = data.frame(lapply(test,function(x) scale(x)))

# Etiquette de valeur de train
lpsa.train = subset(prostate.data,train == T, select=c(lpsa))
# Etiquette de valeur de test
lpsa.test = subset(prostate.data,train == F, select=c(lpsa))

# Plot le resultat => Comparaison entre les etiquettes de prediction et les etiquettes de base de test
#comparison <- data.frame(lpsa.test, test.prediction)
#plot(comparison)
kValues = 3:40
kOpti = 3
minMSE = 100
kMSE.train = rep(0,length(kValues))
kMSE.test = rep(0,length(kValues))

for(kValue in kValues) {
    # Lance Knn et recuperation des étiquettes selon l'algo
    knn.train = knn.reg(train, train, lpsa.train, k = kValue)
    train.prediction = knn.train$pred

    # Transformation in matrix for comparition values
    train.prediction = matrix(train.prediction)
    kMSE.train[kValue-2] = mse(sim=train.prediction,obs=lpsa.train)

    # =====================================================

    # Lance Knn et recuperation des étiquettes selon l'algo
    knn.test = knn.reg(train, test, lpsa.train, k = kValue)
    test.prediction = knn.test$pred

    # Transformation in matrix for comparition values
    test.prediction = matrix(test.prediction)
    kMSE.test[kValue-2] = mse(sim=test.prediction,obs=lpsa.test)

    if(minMSE > kMSE.test[kValue-2]) {
        minMSE = kMSE.test[kValue-2]
        kOpti = kValue
    }
}

resultKGraph.train = data.frame(kValues,kMSE.train)
resultKGraph.test = data.frame(kValues,kMSE.test)
plot(resultKGraph.train,type='b', ylim=range(c(kMSE.train, kMSE.test)),xlab = "", ylab = "")
# second plot  EDIT: needs to have same ylim
par(new = TRUE)
plot(resultKGraph.test,type='b', col='red', ylim=range(c(kMSE.train, kMSE.test)), axes = FALSE, xlab = "k Nearest Neighbor", ylab = "MSE")

readline(prompt="Press [enter] to continue")
cat("Best Value for K: ", kOpti,"\n")
cat("Value of MSE:     ", minMSE,"\n")
par(new = FALSE)