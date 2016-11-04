phoneme = read.table("data/phoneme.data.txt",header=T,sep=",")

# --------------------------------------------- SCALING DATAS ---------------------------------------------------------


phoneme[,1:256] <- as.data.frame(scale(phoneme[,1:256]))

# -------------------------------------------- DATA SEPARATION --------------------------------------------------------

#	Nous utiliserons ici le prinicipe de cross validation :
#	Pour cela nous allons commencer par séparer les valeurs en 2 sous ensembles
#	Un ensemble de validation de taille 1/3 des données
#	Une ensemble d'apprentissage de taille 2/3 des données

#	L'ensemle d'apprentissage va etre divisé aléatoirement en sections
#	Nous choisissons arbitrairement de découper en 10 sections de 300 données chacunes
#	Sur ces 10 sections nous choisissons tour à tour 1 section qui representera 
#	notre ensemble de test. Nous allons donc effectuer la construction de notre modele 10 fois
#	et ensuite choisir celui qui aura les meilleures performances
#	Cette méthode nous permet de nous afrfanchir des 