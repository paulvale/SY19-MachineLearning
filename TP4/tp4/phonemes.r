# LOAD DATA
phoneme = read.table("data/phoneme.data.txt",header=T,sep=",")
# ====
# x.1, x.2, ... x.256, g, speaker
# ====
x <- phoneme$g

y <- phoneme$speaker

# Pour que tu ne sois pas paumé et que tu comprennes ce que j'ai fait, je laisse dans le script pour le moment
# et il te suffit de print a chaque fois

# 1ere question : devons nous entrainer notre modele a reconnaitre les mots d'un certain locuteur 
# ou devons nous reussir a reconnaitre pour n'importe qui et c'est la reconnaissance du mot qui
# est importante ?

# cette prise de position est primordiale, car elle va notamment impliquer une difference au niveau
# de notre division des datas ( test et apprentissage)
# - si on veut bien reconnaitre que les personnes presentes dans nos datas alors il faut diviser notre
# ensemble de maniere a avoir 2/3 des datas d'une personnes dans l'apprentissage et 1/3 de
# ses enregistrements dans le test.
# - si on veut reconnaitre maintenant avant tous l'utilisation des mots dans le discours, 
# alors il est plus interessant de garder 2/3 des personnes et donc tout leur enregistrements
# dans les donnees d'apprentissage et de garder 1/3 des personnes inconnus par notre modele.

# === Description de nos datas ===
# je t'invite deja a lire et relire le doc phoneme.info.txt qui nous ait fournis
# j'avoue que j'ai mis du temps a comprendre mais voila ce que j'ai retenu sur nos datas :
# on fait de la reconnaissance vocale sur des discours, ce que l'on veut retenir nous
# ce n'est que 5 phonemes 'aa', 'sh', 'dck', 'ao', et 'iy' (contrairement a la vraie reconnaissance vocale ou tu as
# beaucoup plus de phonemes a prendre en compte )

# Pour pouvoir creer un modele de reconnaissance de ces phonemes, on a eu tout d'abord
# 50 hommes qui ont fait des discours, on en a retire 4509 petits bouts de discours de 32ms
# avec environ 2 examples de phonemes par speaker.
# 1 frame ( petit bouts de discours ) = 1 phoneme = 512 samples ( on le divise en 512 petits bouts again)

# Voila la repartition de nos differents frames
# aa   ao dcl   iy  sh 
# 695 1022 757 1163 872

# Pour chacun de nos frames, on fait un log-periodogram, technique utilisé dans la reconnaissance vocale
# et ce log-periodogram comprend 256 features
# nos datas 

