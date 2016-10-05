# LOAD DATA
prostate.data = read.table("data/prostate.data.txt")


#Linear regression method
reg.smsa <- lm(prostate.data$lpsa~prostate.data$lcavol+prostate.data$lweight+prostate.data$age+prostate.data$lbph+prostate.data$svi+prostate.data$lcp+prostate.data$gleason+prostate.data$pgg45)

# === Q1 ===
# Les coefficients significativement non nuls sont lcavol, lweigth,svi
# En notation anglo saxonne, "intercept" correspond a l'ordonnee a l'origine
# correspond donc ici au "b" de notre equation f(x) = ax + b

# avoir avoir realise un summary sur notre regression, on peut alors voir
# qu'il y a en effet 3 facteurs determinants sur la prediction de nos valeurs lpsa
# la plus importante est lcavol, mais il y a aussi lweight and svi.
print(summary(reg.smsa))
# les residus (differences entre valeurs originales et valeurs estimees)
# p-val correspond au test de Fisher
# si cette valeur est superieure a 0.05 alors notre regression n'est pas significative.
# Ici, on a une p-value = 2.2e-16 qui est donc bien inferieure a 0.05
# => Notre regression est globalement significative
plot(prostate.data$lcavol,prostate.data$lpsa)
readline(prompt="Press [enter] to continue")

# === Q2 ===
print(confint(reg.smsa,level=0.95))

# === Q3 ===
plot(prostate.data$lpsa,prostate.data$lpsa,col="black")     
abline(a=0,b=1)     
points(prostate.data$lpsa,reg.smsa$fitted.values,pch=19,col="red",cex=0.7)
readline(prompt="Press [enter] to continue")

# === Q4 ===
# Pour les residus bruts :
print("les residus bruts :")
plot(prostate.data$lpsa,reg.smsa$residuals,col="red")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lcavol,reg.smsa$residuals,col="blue")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lweight,reg.smsa$residuals,col="black")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

# A la vue de ces 3 graphiques, on a l'impression que l'on a une evolution croissante des residus 
# en fonction de notre valeur de lpsa, cependant pour les 2 autres variables lweight et lcavol, 
# on n'arrive pas a distinguer une relation entre les residus et ces variables ...

# Pour les residus centre reduit:
residualsStd <- rstandard(reg.smsa)
print("les residus standardises :")
plot(prostate.data$lpsa,residualsStd,col="red")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lcavol,residualsStd,col="blue")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lweight,residualsStd,col="black")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

# Une fois standardise, la relation entre lpsa et les residus est d'autant plus marque,
# en effet, plus la valeur de lpsa est grande plus les residus sont importants et inversement.


# Pour les residus studentise:
residualsStudent <- rstudent(reg.smsa)
print("les residus studentises :")
plot(prostate.data$lpsa,residualsStudent,col="red")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lcavol,residualsStudent,col="blue")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lweight,residualsStudent,col="black")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

# Meme observation que precedement

# === Q5 ===
# Le QQ plot est un graphe qui nous permet de verifier graphiquement la normalite de notre regression
# celui est notamment accessible directement via nos data de reg.smsa

# Comme on peut l'observer, nous avons bel et bien une droite et pouvons donc
# conclure via ce graphique que nos residus suivent bien une loi normale
plot(reg.smsa, which = 2)

# === Q6 ===
#   Pour voir la stabilité du modèle (influence de chaque observation sur les coefficients estimés Bi) il faut utiliser la distance de Cook. Cook mesure l'effet de la suppression de l'observation i sur la prédiction de toutes les valeurs
d1 <- cooks.distance(reg.smsa)
plot(d1)
#   On peut observer sur le graphe que les points ne sont pas abérants (bien inférieur à 1), les valeurs ne sont donc pas trop atypiques. On peut aussi utiliser : plot(reg.smsa,which=4)


# === Q7 ===
#   j'ai repris le script avec seulement 2 predicteurs, les plus significatifs : lcavol et lweight
reg.smsa2 <- lm(prostate.data$lpsa~prostate.data$lcavol+prostate.data$lweight)
print(summary(reg.smsa2))
print(confint(reg.smsa2,level=0.95))
#   On peut voir que les intervalles de confiance réduisent (0.35 avant et 0.27 maintenant pour lcavol, même phénomène pour lweight) les valeurs sont donc plus concentrés autour de la moyenne

plot(prostate.data$lpsa,prostate.data$lpsa,col="black")
abline(a=0,b=1)
points(prostate.data$lpsa,reg.smsa2$fitted.values,pch=19,col="red",cex=0.7)
#   répartition équivalente à l'autre sous ensemble de prédicteur

# Pour les residus bruts :
print("les residus bruts :")
plot(prostate.data$lpsa,reg.smsa2$residuals,col="red")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lcavol,reg.smsa2$residuals,col="blue")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lweight,reg.smsa2$residuals,col="black")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

# Pour les residus centre reduit:
residualsStd2 <- rstandard(reg.smsa2)
print("les residus standardises :")
plot(prostate.data$lpsa,residualsStd2,col="red")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lcavol,residualsStd2,col="blue")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lweight,residualsStd2,col="black")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")


# Pour les residus studentise:
residualsStudent2 <- rstudent(reg.smsa2)
print("les residus studentises :")
plot(prostate.data$lpsa,residualsStudent2,col="red")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lcavol,residualsStudent2,col="blue")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

plot(prostate.data$lweight,residualsStudent2,col="black")
abline(a=0,b=0)
readline(prompt="Press [enter] to continue")

# QQPLOT plus fitté à la courbe linéaire
plot(reg.smsa2, which = 2)
readline(prompt="Press [enter] to continue")

# Distance de COOK - graphe plus ou moins équivalent, pas d'observations trop atypiques
plot(reg.smsa2,which=4)
readline(prompt="Press [enter] to continue")
plot(reg.smsa,which=4)

# Pour résumer, on peut voir que les résultats sont presque équivalent que lorsqu'on avait tous les predicteurs. On peut expliquer cela par le fait qu'on ait pris les predicteurs les plus significatifs sur cette regression lineaire.

#   === Q8 ===
#   Effectuer quelques transformations non linéaires de certains predicteurs




