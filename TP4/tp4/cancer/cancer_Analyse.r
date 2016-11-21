# CLEAN WORKSPACE
rm(list = ls())
library(corrplot)
# ==================================
# ======== Analyse des donnees =====
# ==================================
cancer = read.table("data/r_breast_cancer.data.txt",header=T,sep=",")
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
corrplot(data.cor, type="lower", tl.cex = 0.6)

# on peut voir aussi qu'il y a des correlations importantes entre certains
# de nos predicteurs 
# On pouvait s'en douter, en effet le worst perimeter va influer sur le mean perimeter
# ...

print(summary(data))

hist(label, main = "Repartition of Time Value", xlab="Time")
plot(data$radius_mean,data$radius_worst, main="Correlation entre radius mean et worst", xlab="radius_mean",ylab="radius_worst")



