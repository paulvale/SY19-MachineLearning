# LOAD DATA
prostate.data = read.table("data/prostate.data.txt")


#Linear regression method
reg.smsa <- lm(lpsa ~. - train ,data=prostate.data)

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
# Pour voir la stabilite du modele ( l'influence de chaque obs sur le coeff estime beta_i) il faut utiliser la distance de cook
# Cook mesure l'effet de la suppression de l'obs i sur la prediction 
# d1 <- cooks.distance(reg.smsa)
# plot(d1)
plot(reg.smsa, which = 4)

# On peut observer sur le graphe que nous n'avons pas de points abberants => aucune donnee atypique

# === Q7 ===
# boucle 
# ====
#allCombination = leaps(prostate.data[,1:8],prostate.data[,9])$which
# allCombinationSize = dim(allCombination)

# for(i in 1:allCombinationSize[1]){
  #predictors <- prostate.data[,which(allCombination[i,])]
  #print(prostate.data$lpsa)

  #predictors[,"lpsa"] <- prostate.data[,9]
  #reg.smsaCombination <- lm(prostate.data$lpsa~.-train, data=predictors)
  #plot(reg.smsa)
#}

# ==== 
# observation faite sur predicteurs = lcavol et lweight 
# - les intervalles de confiance se reduisent, on se retrouve plus proche de la moy
# - QQplot est plus fitte a la droite 
# - Cook : plus ou moins identique

# Pour resumer, on peut voir que les resultats sont presques equivalents que lorsque l'on avait tous les predicteurs
# on peut expliquer cela, par la significativite des predicteurs

#Q7
reg1<-lm(lpsa ~. - train -lweight,data=prostate.data)
reg1<- lm(lpsa ~ lcavol+lweight+svi  ,data=prostate.data)
# etc.

# Q8
reg2<- lm(lpsa ~ lcavol+lweight+I(lweight^2) +svi  ,data=prostate.data)
# maintenant, les coefficients de lweight et lweight2 ne sont plus significatifs !