import argparse
from hlm import hetlm
from hlm import hetlmm
import numpy as np
from pysnptools.snpreader import Bed, Pheno
from scipy.stats import chi2, zscore
#import code


####### Output functions ##########

def neglog10pval(x,df):
    return -np.log10(np.e)*chi2.logsf(x,df)

def vector_out(alpha,se,digits=4):
    ## Calculate test statistics
    # t-statistic
    t=alpha/se
    # chi-square statistic
    x2=np.square(t)
    # Create output strings
    if len(alpha.shape)==0:
        pval=neglog10pval(x2,1)
        alpha_print=str(round(alpha,digits))+'\t'+str(round(se,digits))+'\t'+str(round(t,digits))+'\t'+str(round(pval,digits))
    else:
        pvals=[neglog10pval(x,1) for x in x2]
        alpha_print=''
        for i in xrange(0,len(alpha)-1):
            alpha_print+=str(round(alpha[i],digits))+'\t'+str(round(se[i],digits))+'\t'+str(round(t[i],digits))+'\t'+str(round(pvals[i],digits))+'\t'
        i+=1
        alpha_print+=str(round(alpha[i],digits))+'\t'+str(round(se[i],digits))+'\t'+str(round(t[i],digits))+'\t'+str(round(pvals[i],digits))
    return alpha_print

####### Go through chromosome segment and infer AV models ######

def learn_models_chr(args):
    # Get genotype matrix
    genotypes=args.genotypes
    if genotypes.ndim==1:
        chr_length=1
        genotypes=genotypes.reshape(genotypes.shape[0],1)
    else:
        chr_length=genotypes.shape[1]
    n=genotypes.shape[0]
    # Get phenotype
    y=args.phenotype
    if len(y.shape)==2:
        y=y[:,0]
    # Remove phenotype NAs
    y_not_nan=np.logical_not(np.isnan(y))
    if np.sum(y_not_nan)<n:
        y=y[y_not_nan]
        # Remove NAs from genotypes
        genotypes=genotypes[y_not_nan,:]
        n=genotypes.shape[0]
    print(str(n)+' non missing cases from phenotype')
    n=float(n)
    # Get fixed effects
    if 'X' in args:
        # Normalise
        args.X=args.X[y_not_nan,:]
        X_stds=np.std(args.X,axis=0)
        args.X=args.X/X_stds
        # Add intercept
        n_X=args.X.shape[1]+1
        X=np.zeros((n,n_X))
        X[:,0]=np.ones(n)
        X[:,1:n_X]=args.X
        X_names=np.zeros((n_X),dtype='S10')
        X_names[0]='Intercept'
        X_names[1:n_X]=args.X_names
    else:
        X=np.ones((int(n),1))
        n_X=1
        X_names=np.array(['Intercept'])
    if 'V' in args:
        # Normalise
        args.V=args.V[y_not_nan,:]
        V_stds=np.std(args.V,axis=0)
        args.V=args.V/V_stds
        # Add scale
        n_V=args.V.shape[1]+1
        V=np.zeros((n,n_V))
        V[:,0]=np.ones(n)
        V[:,1:n_V]=args.V
        V_names=np.zeros((n_V),dtype='S10')
        V_names[0]='Scale'
        V_names[1:n_V]=args.V_names
    else:
        V=np.ones((int(n),1))
        n_V=1
        V_names=np.array(['Scale'])
    n_pars=n_X+n_V+1
    print(str(n_pars)+' parameters in model')
    # Rescale random effect design matrix if provided
    if 'G' in args:
        args.G = zscore(args.G, axis=0)
        args.G = np.power(args.G.shape[1], -0.5) * args.G
    ######### Initialise output files #######
    ## Output file
    if args.append:
        write_mode='ab'
    else:
        write_mode='wb'
    outfile=open(args.outprefix+'.models.gz',write_mode)
    if not args.append:
        header='SNP_index\tn\tfrequency\tlikelihood\tadd\tadd_se\tadd_t\tadd_pval\tvar\tvar_se\tvar_t\tvar_pval\tav_pval\n'
        outfile.write(header)
    ######### Fit Null Model ##########
    ## Get initial guesses for null model
    print('Fitting Null Model')
    # Optimize null model
    if 'G' in args:
        null_optim= hetlmm.model(y, X, V, args.G).optimize_model(args.h2_init)
    else:
        null_optim= hetlm.model(y, X, V).optimize_model()
    ## Record fitting of null model
    # Get print out for fixed mean effects
    alpha_out=np.zeros((n_X,2))
    alpha_out[:,0]=null_optim['alpha']
    alpha_out[:,1]=null_optim['alpha_se']
    # Rescale
    if n_X>1:
        alpha_out[1:n_X] = alpha_out[1:n_X]/X_stds
    if not args.append:
        np.savetxt(args.outprefix + '.null_mean_effects.txt',
                   np.hstack((X_names.reshape((n_X, 1)), np.array(alpha_out, dtype='S20'))),
                   delimiter='\t', fmt='%s')
    # variance effects
    beta_out=np.zeros((n_V,2))
    beta_out[0:n_V,0]=null_optim['beta']
    beta_out[0:n_V,1]=null_optim['beta_se']
    # Rescale
    if n_V>1:
        beta_out[1:n_X] = beta_out[1:n_X]/V_stds
    if not args.append:
        np.savetxt(args.outprefix + '.null_variance_effects.txt',
                   np.hstack((V_names.reshape((n_V, 1)), np.array(beta_out, dtype='S20'))),
                   delimiter='\t', fmt='%s')
    if 'G' in args:
        if not args.append:
            np.savetxt(args.outprefix + '.null_h2.txt',
                       np.array([null_optim['h2'], null_optim['h2_se']], dtype='S20'),
                       delimiter='\t', fmt='%s')
    if not args.fit_covariates:
        # Residual phenotype
        y=y-X.dot(null_optim['alpha'])
        # Reformulate fixed_effects
        X=np.ones((int(n),1))
        n_X=1
        # Rescaled residual phenotype
        D_null_sqrt=np.exp(0.5*V.dot(null_optim['beta']))
        y=y/D_null_sqrt
        # Reformulate fixed variance effects
        V=np.ones((int(n),1))
        n_V=1
    ## Loop through loci
    for loc in xrange(0,chr_length):
        # Filler for output if locus doesn't pass thresholds
        additive_av_out='NaN\tNaN\tNaN\tNaN'
        variance_out='NaN\tNaN\tNaN\tNaN'
        likelihood=np.nan
        allele_frq=np.nan
        av_pval=np.nan
        # Get test genotypes
        test_gts=genotypes[:,loc]
        # Find missingness and allele freq
        test_gt_not_na=np.logical_not(np.isnan(test_gts))
        n_l=np.sum(test_gt_not_na)
        missingness = 100.0 * (1 - float(n_l) / n)
        if missingness<args.max_missing:
            test_gts=test_gts[test_gt_not_na]
            test_gts = test_gts.reshape((test_gts.shape[0], 1))
            allele_frq=np.mean(test_gts)/2
            if allele_frq>0.5:
                allele_frq=1-allele_frq
            if allele_frq>args.min_maf:
                # Remove missing data rows
                y_l=y[test_gt_not_na]
                X_l=X[test_gt_not_na,:]
                V_l=V[test_gt_not_na,:]
                X_l=np.hstack((X_l,test_gts))
                V_l=np.hstack((V_l,test_gts))
                print('Fitting locus AV model for locus '+str(loc))
                if 'G' in args:
                    G_l = args.G[test_gt_not_na, :]
                    av_optim = hetlmm.model(y_l, X_l, V_l, G_l).optimize_model(null_optim['h2'])
                else:
                    av_optim= hetlm.model(y_l, X_l, V_l).optimize_model()
                #if av_optim['success']:
                if True:
                    # Likelihood
                    likelihood=-0.5*(av_optim['likelihood']+n_l*np.log(2*np.pi))
                    # Mean effect of locus
                    additive_av_out=vector_out(av_optim['alpha'][n_X],av_optim['alpha_se'][n_X],6)
                    # Variance effect of locus
                    variance_out=vector_out(av_optim['beta'][n_V],av_optim['beta_se'][n_V],6)
                    av_pval=neglog10pval((av_optim['alpha'][n_X]/av_optim['alpha_se'][n_X])**2+(av_optim['beta'][n_V]/av_optim['beta_se'][n_V])**2,2)
                else:
                    print('Failed to converge for for locus '+str(loc))
        outfile.write(str(args.start+loc) + '\t'+str(n_l)+'\t'+ str(allele_frq)+'\t'+str(likelihood)+'\t'+additive_av_out+'\t'+variance_out+'\t'+str(round(av_pval,6))+'\n')
    outfile.close()
    return

def id_dict_make(ids):
    if not type(ids)==np.ndarray:
        raise(ValueError('Unsupported ID type: should be numpy nd.array'))
    id_dict={}
    for id_index in xrange(0,len(ids)):
        id_dict[tuple(ids[id_index,:])]=id_index
    return id_dict


if __name__ == "__main__":
    ######## Parse Arguments #########
    parser=argparse.ArgumentParser()
    parser.add_argument('genofile',type=str,help='Path to genotypes in BED format')
    parser.add_argument('start',type=int,help='Index of SNP in genofile from which to start computing test stats')
    parser.add_argument('end',type=int,help='Index of SNP in genofile at which to finish computing test stats')
    parser.add_argument('phenofile',type=str,help='Location of the phenotype file in PLINK format')
    parser.add_argument('outprefix',type=str,help='Location to output csv file with association statistics')
    parser.add_argument('--mean_covar',type=str,help='Location of mean covariate file in PLINK format (default None)',
                        default=None)
    parser.add_argument('--var_covar',type=str,help='Locaiton of variance covariate file in PLINK format (default None)',
                        default=None)
    parser.add_argument('--fit_covariates',action='store_true',
                        help='Fit covariates for each locus. Default is to fit for null model and project out (mean) and rescale (variance)',
                        default=False)
    parser.add_argument('--random_gts',type=str,help='Location of the BED file with the genotypes of the SNPs that random effects should be modelled for',default=None)
    parser.add_argument('--h2_init',type=float,help='Initial value for variance explained by random effects (default 0.05)',
                        default=0.05)
    parser.add_argument('--phen_index',type=int,help='If phenotype file contains multiple phenotypes, which column to choose (default 1, first)',
                        default=1)
    parser.add_argument('--min_maf',type=float,help='Minimum minor allele frequency (default 0.05)',default=0.05)
    parser.add_argument('--missing_char',type=str,help='Missing value string in phenotype file (default NA)',default='NA')
    parser.add_argument('--max_missing',type=float,help='Maximum percent of missing genotype calls (default 5)',default=5)
    parser.add_argument('--append',action='store_true',default=False,help='Append results to existing output file')

    args=parser.parse_args()

    test_chr=Bed(args.genofile)
    # select subset to test
    test_chr=test_chr[:,args.start:args.end].read()
    args.genotypes=test_chr.val
    print('Number of test loci: '+str(args.genotypes.shape[1]))
    # Get sample ids
    geno_ids=test_chr.iid
    # Get phenotype file
    pheno=Pheno(args.phenofile,iid_if_none=geno_ids,missing=args.missing_char).read()
    args.phenotype=np.array(pheno.val)
    if args.phenotype.ndim==1:
        pheno_noNA=args.phenotype[np.logical_not(np.isnan(args.phenotype))]
        args.phenotype=args.phenotype/pheno_noNA.std()
    elif args.phenotype.ndim==2:
        args.phenotype=args.phenotype[:,args.phen_index-1]
        pheno_noNA=args.phenotype[np.logical_not(np.isnan(args.phenotype))]
        args.phenotype=args.phenotype/pheno_noNA.std()
    else:
        raise(ValueError('Incorrect dimensions of phenotype array'))
    print('Number of phenotype observations: '+str(args.phenotype.shape[0]))
    # Match IDs with geno IDs
    pheno_ids=np.array(pheno.iid)
    pheno_id_dict=id_dict_make(pheno_ids)
    pheno_id_match=np.array([pheno_id_dict[tuple(x)] for x in geno_ids])
    args.phenotype=args.phenotype[pheno_id_match]

    ## Get covariates
    if not args.mean_covar==None:
        mean_covar_f=Pheno(args.mean_covar,iid_if_none=geno_ids,missing=args.missing_char).read()
        args.X=mean_covar_f.val
        args.X_names=np.array(mean_covar_f._col,dtype='S20')
        # Match with geno_ids
        mean_ids_dict=id_dict_make(np.array(mean_covar_f.iid))
        X_id_match=np.array([mean_ids_dict[tuple(x)] for x in geno_ids])
        args.X=args.X[X_id_match,:]

    if not args.var_covar==None:
        var_covar_f=Pheno(args.var_covar,iid_if_none=geno_ids,missing=args.missing_char).read()
        args.V=var_covar_f.val
        args.V_names=np.array(var_covar_f._col,dtype='S20')
        # Match with geno_ids
        var_ids_dict=id_dict_make(np.array(var_covar_f.iid))
        V_id_match=np.array([var_ids_dict[tuple(x)] for x in geno_ids])
        args.V=args.V[V_id_match,:]

    # Get random effect loci
    if args.random_gts is not None:
        random_gts_f=Bed(args.random_gts).read()
        args.G=random_gts_f.val
        # Check for NAs
        random_isnan=np.isnan(args.G)
        random_gts_NAs=np.sum(random_isnan,axis=0)
        gts_with_obs=list()
        if np.sum(random_gts_NAs)>0:
            print('Mean imputing missing genotypes in random effect design matrix')
            for i in xrange(0,args.G.shape[1]):
                if random_gts_NAs[i]<args.G.shape[0]:
                    gts_with_obs.append(i)
                    if random_gts_NAs[i]>0:
                        gt_mean=np.mean(args.G[np.logical_not(random_isnan[:,i]),i])
                        args.G[random_isnan[:,i],i]=gt_mean
            # Keep only columns with observations
            args.G=args.G[:,gts_with_obs]
        print(str(int(args.G.shape[1]))+' loci in random effect')

    learn_models_chr(args)