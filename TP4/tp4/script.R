f1 <- function(tab,mot)
{
	n <- length(tab)
	i=0;
	res <- vector("numeric", length=n)
	for(i in 1:n)
	{
		if(tab[i]==mot)
		{
			res[i]=1;
		}
		else{
			res[i]=0;
		}
	}
	res
}

distXY <- function(X, Y)
{
	nx <- dim(X)[1]
	ny <- dim(Y)[1]
	h.x <- rowSums(X^2)
	h.y <- rowSums(Y^2)
	ones.x <- rep(1,nx)
	ones.y <- rep(1,ny)
	D2xy <- h.x%*%t(ones.y) - 2*X %*% t(Y) + ones.x%*%t(h.y)
}

kppv <- function(Xtrain,Ztrain,k,Xtst)
{
	etiq=matrix(nrow=0,ncol=1);
	distances<- distXY(as.matrix(Xtst), as.matrix(Xtrain));
	class=rep('',5);
	plus_proches = c(0,k)
	plus_proches2 = c(0,k)
	res = c('aa','ao','dcl','sh','iy');

	for(i in 1:dim(distances)[1])
	{
		kclass=rep(0,k);
		dimDistances <- dim(distances)[2]
		for (j in 1:k)
		{
			plus_proches[j] <- distances[i,1];
			for(it in 1:dimDistances)
			{
				if(distances[i,it]<plus_proches[j])
				{
					
					indicateur <- 0;
					for(it2 in 1:length(plus_proches2))
					{
						if(plus_proches2[it2]==it)
						{
							indicateur <- 1;
						}
					}
					if(indicateur ==0)
					{
						plus_proches[j]<-distances[i,it];
						plus_proches2[j]<-it
					}
					
				}
			}
		}

		#associagion des etiquettes aux plus proches vosins trouvés
		for (j in 1:k)
		{
			kclass[j] <- as.character(Ztrain[plus_proches2[j]])
		}

		class=c(length(which(kclass=='aa')),length(which(kclass=='ao')),length(which(kclass=='dcl')),length(which(kclass=='sh')),length(which(kclass=='iy')))
		max<-class[1]
		result<-res[1]
		for(h in 1:5)
		{
			if(class[h]>max)
			{
				max <- class[h];
				result <- res[h]
			}
		}
		etiq <- rbind(etiq,c(result));
	}
	etiq;
}