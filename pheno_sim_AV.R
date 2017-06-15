args=commandArgs(trailingOnly=T)

v=args[1]

h2=args[2]

library(rhdf5)

g=h5read('/well/donnelly/ukbiobank_project_8874/ay/linear_variance/new_simulations/artificial_gt_0.5.hdf5','genotypes')

g=g[,1]

add_effect=sqrt(2*as.numeric(h2))

y=replicate(1000,sapply(g,function(x) rnorm(1,add_effect*x,exp(as.numeric(v)*x))))

h5createFile(paste('/well/donnelly/ukbiobank_project_8874/ay/linear_variance/new_simulations/av_vs_a/h2_',h2,'_v_',v,'.hdf5',sep=''))
h5write(y,paste('/well/donnelly/ukbiobank_project_8874/ay/linear_variance/new_simulations/av_vs_a/h2_',h2,'_v_',v,'.hdf5',sep=''),'phenotypes')
h5write(as.character(1:10^5),paste('/well/donnelly/ukbiobank_project_8874/ay/linear_variance/new_simulations/av_vs_a/h2_',h2,'_v_',v,'.hdf5',sep=''),'sample_id')