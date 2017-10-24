require(BEDMatrix)
######### SIMULATE GAMMA DISTRIBUTED PHENOTYPE FROM WHOLE CHR ########
#new_simulations/mean_and_var/gamma
loc_sim=function(g,h2,v){
  if (h2==0 & v==0){ 
    s=1.8
    gamma_mean=5
    y=rgamma(dim(g)[1],gamma_mean/s,scale=s)
  } else {
  add_effects=rnorm(dim(g)[2]-1)
  add=as.matrix(g[,-1])%*%as.matrix(add_effects)
  print(paste('h2=',h2,sep=''))
  add=sqrt(0.995)*scale(add)
  add=add+sqrt(2*0.005)*g[,1]
  gamma_mean=ceiling(abs(min(add)))
  s=(1-h2)/(gamma_mean*h2)
  print(s)
  g1=scale(g[,1],scale=F)
  y=sapply(1:length(add),function(x) rgamma(1,exp(-v*g1[x])*(gamma_mean+add[x])/s,scale=s*exp(v*g1[x])))}
  return(y)
}


make_inv_norm_y=function(y){
ecdf_y=ecdf(y)
return(qnorm(ecdf_y(y)*(length(y)/(length(y)+1))))
}

# Causal sample
setwd('~/PycharmProjects/hlm/tests/')
g_causal=read.table('genotypes.txt',header=F,colClasses = rep('integer',1000))

y=loc_sim(g_causal,0.10,0.05)
inv_norm_y=make_inv_norm_y(y)

fam=read.table('test.fam')
write.table(data.frame(fam[,1:2],inv_norm_y),'test_phenotype.fam',quote=F,row.names=F,col.names=F)

######### Analysis #######
cbind_pvals=function(combined,colnames,dfs){
  names(dfs)=colnames
  for (colname in colnames){
    combined=cbind(combined,
                   -log10(pchisq(combined[,colname],dfs[colname],lower.tail=F)))
    dimnames(combined)[[2]][dim(combined)[2]]=paste(colname,'pval',sep='_')}
  return(combined)
}

results=read.table('test.models.gz',header=T)

require(MASS)
r_var_mean=rlm(var~0+add,data=results)

mean_noise=mean(results$add_se^2,na.rm=T)
noise_adjustment=1+mean_noise/(var(results$add,na.rm=T)-mean_noise)

r_av=r_var_mean$coefficients[1]*noise_adjustment

results$dispersion=results$var-r_av*results$add
results$dispersion_se=sqrt(results$var_se^2+(r_av^2)*results$add_se^2)
results$dispersion_t=results$dispersion/results$dispersion_se
results$dispersion_pval=-log10(pchisq(results$dispersion_t^2,1,lower.tail=F))

causal=read.table(paste(dir,'causal_gts.txt',sep=''),header=T)

indices=1:dim(results)[1]
causal_indices=c()
for (i in 1:dim(causal)[1]){
  causal_indices=c(causal_indices,indices[results$chr==causal[i,1] & results$SNP_index==(causal[i,2]-1)])
}
dispersion=rbind(dispersion,results[causal_indices[1],])
causal_results=rbind(causal_results,results[causal_indices[-1],])
null_results=rbind(null_results,results[-causal_indices,])
print(dir)
write.table(results,paste(dir,'results.txt',sep=''),quote=F,row.names=F)

save.image('results.RData')

# Form random effects

#directory='/well/donnelly/ukbiobank_project_8874/ay/linear_variance/new_simulations/mixed_model/'
rep=paste('rep',5,sep='_')
directory=paste(getwd(),rep,sep='/')
results=read.table(paste(directory,'/results.txt',sep=''),header=T)
nrnd=1000
#load('/well/donnelly/ukbiobank_project_8874/ay/log_bmi/collated_pvals.RData')
require(rhdf5)
#load(paste(directory,'llrs_collated.RData',sep=''))


results=results[order(-results[,'mean_llr']),]

# Get sample size
chr=results[1,'chr']
index=results[1,'SNP_index']+1
chr_file=paste('/well/donnelly/ukbiobank_project_8874/ay/linear_variance/new_simulations/mixed_model/',rep,'/artificial_chr',chr,'.hdf5',sep='')
gts=h5read(chr_file,'genotypes',index=list(NULL,index))
sample_id=h5read(chr_file,'sample_id')
n=length(gts)
# Array to fill

condition_gts=matrix(NA,nrow=n,ncol=nrnd)
condition_gts_locations=matrix(NA,nrow=nrnd,ncol=2)

filled=0
list_index=1
while (filled<nrnd){
  gt_na=gts<0
  missingness=sum(gt_na)/n
  print(paste('Missingness:',missingness))
  print(paste('chr',chr,sep='='))
  if (missingness<0.05){
    # Check frequency
    gt_mean=mean(gts[!gt_na])
    gt_frq=gt_mean/2
    if (gt_frq>0.5){gt_frq=1-gt_frq}
    print(paste('Frequency:',gt_frq))
    if (gt_frq>0.05){
      # Mean impute
      gts[gt_na]=gt_mean
      # Check correlation with previously included gts
      if (filled>0){
        cor_gt=0
        on_chr=condition_gts_locations[,1]==chr
        if (sum(on_chr,na.rm=T)>0){
          cor_gt=cor(gts,condition_gts[,on_chr & !is.na(on_chr)])}
        if (max(abs(cor_gt))<0.9){
          filled=filled+1
          print(paste('filled',filled))
          condition_gts[,filled]=gts
          condition_gts_locations[filled,]=c(chr,index)
        }
      } else {
        filled=filled+1
        print(paste('filled',filled))
        condition_gts[,filled]=gts
        condition_gts_locations[filled,]=c(chr,index)}
    }
  }
  list_index=list_index+1
  chr=results[list_index,'chr']
  index=results[list_index,'SNP_index']+1
  H5close()
  chr_file=paste('/well/donnelly/ukbiobank_project_8874/ay/linear_variance/new_simulations/mixed_model/',rep,'/artificial_chr',chr,'.hdf5',sep='')
  gts=h5read(chr_file,'genotypes',index=list(NULL,index))
  sample_id_chr=h5read(chr_file,'sample_id')
  gts=gts[match(sample_id,sample_id_chr)]
  print(list_index)
}
condition_gts=scale(condition_gts)

for (exclude_chr in 1:22){
  condition_gts_chr=condition_gts[,condition_gts_locations[,1]!=exclude_chr]
  h5out=paste(directory,'random_effect_',exclude_chr,'.hdf5',sep='')
  h5createFile(h5out)
  h5write(condition_gts_chr,h5out,
          'genotypes')
  h5write(sample_id,h5out,'sample_id')
  H5close()
  print(exclude_chr)}


##### Mixed model analysis 

cbind_pvals=function(combined,colnames,dfs){
  names(dfs)=colnames
  for (colname in colnames){
    combined=cbind(combined,
                   -log10(pchisq(combined[,colname],dfs[colname],lower.tail=F)))
    dimnames(combined)[[2]][dim(combined)[2]]=paste(colname,'pval',sep='_')}
  return(combined)
}

dirs=c(sapply(c(1:10),function(x) paste('rep_',x,'/rnd/',sep='')))
null_results=c()
causal_results=c()
dispersion=c()
r_avs=c()

for (dir in dirs){
  if (dir==dirs[1]){
    results=read.table(paste(dir,'results.txt',sep=''),header=T)
  } else {
  i=1
  results=rbind(cbind(i,read.table(paste(dir,'chr.',sprintf('%02d',i),'.0.models.gz',sep=''),header=T,nrow=500)),
                cbind(i,read.table(paste(dir,'chr.',sprintf('%02d',i),'.1.models.gz',sep=''),header=T,nrow=500)))
  for (i in 2:22){
    results=rbind(results,rbind(cbind(i,read.table(paste(dir,'chr.',sprintf('%02d',i),'.0.models.gz',sep=''),header=T,nrow=500)),
                                cbind(i,read.table(paste(dir,'chr.',sprintf('%02d',i),'.1.models.gz',sep=''),header=T,nrow=500))))
  }
  dimnames(results)[[2]][1]='chr'
  
  
  require(MASS)
  r_var_mean=rlm(var_effect~0+mean_effect,data=results,weights=results$var_effect_se^(-2),wt.method='inv.var')
  mean_noise=mean(results$mean_effect_se^2,na.rm=T)
  noise_adjustment=1+mean_noise/(var(results$mean_effect,na.rm=T)-mean_noise)
  
  r_av=r_var_mean$coefficients[1]*noise_adjustment
  r_avs=c(r_avs,r_av)
  
  results$dispersion_effect=results$var_effect-r_av*results$mean_effect
  results$dispersion_effect_se=sqrt(results$var_effect_se^2+(r_av^2)*results$mean_effect_se^2)
  results$dispersion_llr=(results$dispersion_effect/results$dispersion_effect_se)^2
  
  results$av_llr=results$mean_llr+results$var_llr
  
  results=cbind_pvals(results,c('mean_llr','var_llr','av_llr','dispersion_llr'),c(1,1,2,1))
  write.table(results,paste(dir,'results.txt',sep=''),quote=F,row.names=F)
  }
  causal=read.table(paste(dir,'../causal_gts.txt',sep=''),header=T)
  
  indices=1:dim(results)[1]
  causal_indices=c()
  for (i in 1:dim(causal)[1]){
    causal_indices=c(causal_indices,indices[results$chr==causal[i,1] & results$SNP_index==(causal[i,2]-1)])
  }
  dispersion=rbind(dispersion,results[causal_indices[1],])
  causal_results=rbind(causal_results,results[causal_indices[-1],])
  null_results=rbind(null_results,results[-causal_indices,])
  print(dir)
}

save.image('results_rnd.RData')
