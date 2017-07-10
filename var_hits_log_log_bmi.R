# Title     : TODO
# Objective : TODO
# Created by: ay
# Created on: 20/06/2017

vhits=read.csv('/well/donnelly/glmm/mimix_paper/log_bmi_var_hits_mixed_model_reduced_resubmission.csv',header=T)

for (i in 1:dim(vhits)[1]){
    v=vhits[i,]
    qprefix=paste('qsub -q short.qc  -v OMP_NUM_THREADS=1 -o /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/log_log_bmi/out/',
                    v[,'rsid'],' -e /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/log_log_bmi/err/',v[,'rsid'],sep='')
    command=paste(qprefix,' linear_heteroskedastic_mixed_model.py /well/donnelly/ukbiobank_project_8874/ay/genotypes/chr.',v[,'chr'],
            '.hdf5 ',v[,'SNP_index'],' ',v[,'SNP_index']+1,
            ' /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/log_log_bmi/log_log_bmi.hdf5 /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/log_bmi/random_effect_',
                v[,'chr'],'.hdf5 /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/log_log_bmi/',
            v[,'rsid'],' --mean_covar /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/covariates.hdf5 --variance_covar /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/covariates.hdf5',
            sep='')
    print(command)
    system(command)
}

llbmi_var_effects=rep(NA,dim(vhits)[1])

for (i in 1:dim(vhits)[1]){
    v=vhits[i,]
    r=read.table(paste('/well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/log_log_bmi/',v[,'rsid'],'.models.gz',sep=''),header=T)
    llbmi_var_effects[i]=r$var_effect}

vhits=cbind(vhits,llbmi_var_effects)

for (i in 1:dim(vhits)[1]){
    v=vhits[i,]
    qprefix=paste('qsub -q short.qc  -v OMP_NUM_THREADS=1 -o /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/bmi/out/',
                    v[,'rsid'],' -e /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/bmi/err/',v[,'rsid'],sep='')
    command=paste(qprefix,' linear_heteroskedastic_mixed_model.py /well/donnelly/ukbiobank_project_8874/ay/genotypes/chr.',v[,'chr'],
            '.hdf5 ',v[,'SNP_index'],' ',v[,'SNP_index']+1,
            ' /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/bmi/bmi.hdf5 /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/log_bmi/random_effect_',
                v[,'chr'],'.hdf5 /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/bmi/',
            v[,'rsid'],' --mean_covar /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/covariates.hdf5 --variance_covar /well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/covariates.hdf5',
            sep='')
    print(command)
    system(command)
}

bmi_var_effects=rep(NA,dim(vhits)[1])

for (i in 1:dim(vhits)[1]){
    v=vhits[i,]
    r=read.table(paste('/well/donnelly/ukbiobank_project_8874/ay/linear_variance/british/bmi/',v[,'rsid'],'.models.gz',sep=''),header=T)
    bmi_var_effects[i]=r$var_effect}

vhits=cbind(vhits,bmi_var_effects)