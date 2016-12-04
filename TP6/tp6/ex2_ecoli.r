# LOAD DATA
library(mclust)
ecoli <- read.table("data/ecoli.txt")
ecoli[1] <- NULL

summary(ecoli)
plot(ecoli, col=c("red", "blue", "green", "cadetblue", "antiquewhite", "brown", "darkgray", "gold")[ecoli$V9])

# => Pour tester tous les modeles on fait alors
# Mais c'est long ... env. 
# En quoi dans le cours on a reussi a voir avec les plots le meilleur modele ...
# Juste en faisant le summary ca suffit pour dire que c'est le meilleur modele .?
ecoliMclust <- Mclust(data = ecoli)
print(summary(ecoliMclust))

# On va donc reessayer mais avec cette fois ci que 8 groupes 
ecoliMclust2 <- Mclust(data = ecoli, G=8)
print(summary(ecoliMclust2))

# Resultats finaux :
#                                                   Reel  |  sans info | avec info
# cp  (cytoplasm)                                    143  |     143    |     143
# im  (inner membrane without signal sequence)        77  |     9      |      13
# pp  (perisplasm)                                    52  |     68     |      64
# imU (inner membrane, uncleavable signal sequence)   35  |     11     |       7
# om  (outer membrane)                                20  |     34     |      33
# omL (outer membrane lipoprotein)                     5  |     19     |      19
# imL (inner membrane lipoprotein)                     2  |     52     |       5
# imS (inner membrane, cleavable signal sequence)      2  |            |      52

# On a du coup beaucoup plus d'erreurs cette fois ci dans cette excercice

# Question prof : 
# Comment peut-on savoir que leur estimation du 1 correspond a la classe cp et pas a une autre classe ?



