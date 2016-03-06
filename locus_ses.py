import os
chrs=[6,18,3,3,4,6,11,1,16,2,19,11,10]
snp_indices=[21406,6197,50432,36987,15499,20974,22920,56517,15182,197,20460,11420,32255]
rsids=['rs2814992','rs1652376','rs61587156','rs957919','rs6831020',
       'rs9469488','rs4441044','rs2785980','rs1421085','rs6548238','rs1800437','rs6265','rs7903146']

n_loci=len(chrs)

geno_prefix='/well/donnelly/ukbiobank_project_8874/ay/genotypes/chr.'
geno_suffix='.hdf5'

sample='british'
lv='/well/donnelly/ukbiobank_project_8874/ay/linear_variance/'+sample+'/'

for i in xrange(0,n_loci):
    command='python /well/donnelly/glmm/hlmm/linear_heteroskedastic_mixed_model.py '
    command+=geno_prefix+str(chrs[i])+geno_suffix+' '
    command+=str(snp_indices[i])+' '+str(snp_indices[i]+1)+' '
    command+=lv+'log_bmi/log_bmi.hdf5 '
    command+=lv+'log_bmi/random_effect_'+str(chrs[i])+'.hdf5 '
    command+=lv+'log_bmi/hits/'+rsids[i]+' '
    command+='--mean_covar '+lv+'covariates.hdf5 '
    command+='--variance_covar '+lv+'covariates.hdf5 '
    command+='--full_cov'
    print(command)
    os.system(command)