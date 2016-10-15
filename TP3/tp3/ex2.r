# Donnee de l'enonce
# ==================
library(MASS)

pi1 <- 0.5
pi2 <- 0.5

mu1 <- matrix(0, nrow=3) 
mu2 <- matrix(1, nrow=3)

sigma1 = diag(3)
sigma2 = 0.8*sigma1
nvalues <- c(30, 100, 1000, 10000)
erreur <- matrix(0, nrow=4, ncol=2)
nTest <- 10000

for (j in 1:length(nvalues)) {
  # nombre d'indivs total = 10000 de test + nvalues[j] d'app
  n <- nvalues[j] + nTest
  # initialisation de la matrice de data
  data <- matrix(0, nrow=n, ncol=4)
  class <- rbinom(n, 1, 0.5)
  for (i in 1:length(class)){
    if (class[i] == 0){
      l <- mvrnorm(1, mu2, sigma2)
    }
    else {
      l<-mvrnorm(1, mu1, sigma1)
    }
    data[i,] <- c(l, class[i])
  }
  
  # Separation des datas en donnees de test et d'app
  test <- data[1:10000,]
  test <- as.data.frame(test)
  
  app <- data[10001:n,]
  app <- as.data.frame(app)
  
  # LDA sur les datas
  lda.app <- lda(V4~. , data=app)
  pred.lda <- predict(lda.app, newdata=test)
  perf.lda <-table(test$V4, pred.lda$class)
  
  # QDA sur les datas
  qda.app <- qda(V4~. , data=app)
  pred.qda <- predict(qda.app, newdata=test)
  perf.qda <-table(test$V4, pred.qda$class)
  
  # Stockage du taux d'erreurs de chacun
  erreur[j,1] <- (1-sum(diag(perf.lda))/10000)*100
  erreur[j,2] <- (1-sum(diag(perf.qda))/10000)*100
}
erreur <- as.data.frame(erreur)
rownames(erreur) <- c("n = 30","n = 100", "n = 1000", "n = 10,000")
colnames(erreur) <- c("ADL","ADQ")
print(erreur)

# Comme on peut le voir, plus le nombre d'inds d'apprentissage est grand et plus notre ADQ
# sera performant. Cependant, pour l'ADL nous avons des resultat qui se montrent
# plus aléatoires.
# ====
# Avec un nombre de valeur d'apps tres grand, on va donc preferer l'ADQ qui nous
# donnera un taux d'erreur plus faible que l'ADL.



