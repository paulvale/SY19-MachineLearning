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
	return res
}