### Simulates a polygenic phenotype according to a Gamma model ###
## All the loci other than the first are given effects on the mean of the trait
## that also effect the variance through the mean-variance relation of the Gamma distribution
## the first variant is also given a dispersion effect
loc_sim=function(g,h2,v){
  # g is a matrix of genotypes
  # h2 is the heritability of the trait
  # v is the dispersion effect of the first variant
  if (h2==0 & v==0){
    s=1.8
    gamma_mean=5
    y=rgamma(dim(g)[1],gamma_mean/s,scale=s)
  } else {
    # Simulate additive effects
  add_effects=rnorm(dim(g)[2]-1)
  add=as.matrix(g[,-1])%*%as.matrix(add_effects)
  add=sqrt(0.995)*scale(add)
  add=add+sqrt(2*0.005)*g[,1]
    # Mean of gamma distribution
  gamma_mean=ceiling(abs(min(add)))
    # Scale parameter of gamma distribution
  s=(1-h2)/(gamma_mean*h2)
  g1=scale(g[,1],scale=F)
    # Simulate phenotype
  y=sapply(1:length(add),function(x) rgamma(1,exp(-v*g1[x])*(gamma_mean+add[x])/s,scale=s*exp(v*g1[x])))}
  return(y)
}

# Function to perform inverse normal transformation based on empirical CDF
make_inv_norm_y=function(y){
ecdf_y=ecdf(y)
return(qnorm(ecdf_y(y)*(length(y)/(length(y)+1))))
}

## Read genotypes
require(BEDMatrix)
g=cbind(as.matrix(BEDMatrix('test')),as.matrix(BEDMatrix('random')))

# Simulate phenotype
y=loc_sim(g,0.10,0.05)
# perform inverse normal transformation
inv_norm_y=make_inv_norm_y(y)
# Read in IDs
fam=read.table('test.fam',stringsAsFactors=F)
# Write phenotype
write.table(data.frame(fam[,1:2],inv_norm_y),'test_phenotype.fam',quote=F,row.names=F,col.names=F)