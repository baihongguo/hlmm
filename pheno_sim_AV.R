#!/apps/well/R/3.1.3/bin/Rscript
args=commandArgs(trailingOnly=T)

v=args[1]

h2=args[2]

h2_d=args[3]

gvar=args[4]

outprefix=args[5]

dom_gt_make=function(g,f){
    if (g==0){
        return(f^2)
    } else if (g==1){
    return(-f*(1-f))
} else if (g==2){
        return((1-f)^2)
    }
}

library(rhdf5)

g=h5read('/well/donnelly/ukbiobank_project_8874/ay/linear_variance/new_simulations/artificial_gt_0.5.hdf5','genotypes')

g=g[,1]

add_effect=sqrt(2*as.numeric(h2))

f=mean(g)/2

dom_gts=sapply(g,dom_gt_make,f)

dom_effect=sqrt(as.numeric(h2_d))/(f*(1-f))



y=replicate(1000,sapply(g,function(x) rnorm(1,add_effect*x+dom_effect*dom_gts,exp(as.numeric(v)*x+as.numeric(gvar)*dom_gts))))

outfile=paste(outprefix,'/h2_',h2,'_v_',v,'_h2d_',h2_d,'_gvar_',gvar,'.hdf5',sep='')
h5createFile(outfile)
h5write(y,outfile,'phenotypes')
h5write(as.character(1:10^5),outfile,'sample_id')