
prostate.data = read.table("prostate.data.txt")
install.packages('FNN')
install.packages("automap")
install.packages("spacetime")
install.packages("sp")
install.packages("zoo")
install.packages("hydroGOF")

# ===================================
# REGRESSION PAR LA METHODE DES KPPVS
# ===================================

plot(prostate.data,col='blue',pch=20)

# Valeur de train sans etiquette
train = subset(prostate.data,train == T, select=c(lcavol,lweight,lbph,age))
# Valeur de test sans etiquette
test = subset(prostate.data,train == F, select=c(lcavol,lweight,lbph,age))

# Etiquette de valeur de train
lpsa.train = subset(prostate.data,train == T, select=c(lpsa))
# Etiquette de valeur de test
lpsa.test = subset(prostate.data,train == F, select=c(lpsa))

# Plot le resultat => Comparaison entre les etiquettes de prediction et les etiquettes de base de test
#comparison <- data.frame(lpsa.test, test.prediction)
#plot(comparison)
kValues = 3:30
kMSE = rep(0,length(kValues))

for(kValue in kValues) {
    # Lance Knn et recuperation des étiquettes selon l'algo
    knn = knn.reg(train, test, lpsa.train, k = kValue)
    test.prediction = knn$pred

    # Transformation in matrix for comparition values
    test.prediction = matrix(test.prediction)
    kMSE[kValue-2] = mse(test.prediction,lpsa.test)
}

resultKGraph = data.frame(kValues,kErrors)
plot(resultKGraph,type='b')

