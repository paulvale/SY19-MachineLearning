
prostate.data = read.table("prostate.data.txt")
plot(prostate.data,col='blue',pch=20)
install.packages('FNN')


# Valeur de train sans etiquette
train = subset(prostate.data,train == T, select=c(lcavol,lweight,lbph,age))
# Valeur de test sans etiquette
test = subset(prostate.data,train == F, select=c(lcavol,lweight,lbph,age))

# Etiquette de valeur de train
lpsa.train = subset(prostate.data,train == T, select=c(lpsa)
# Etiquette de valeur de test
lpsa.test = subset(prostate.data,train == F, select=c(lpsa))

# Lance Knn et recuperation des étiquettes selon l'algo
knn = knn.reg(train, test, lpsa.train, k = 5)
test.prediction = knn$pred

# Plot le resultat => Comparaison entre les etiquettes de prediction et les etiquettes de base de test
comparison <- data.frame(lpsa.test, test.prediction)
plot(comparison)