import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import argparse, code
from pysnptools.snpreader import Bed, Pheno

class hlm_model(object):
    """
    A heteroskedastic linear model
    """
    def __init__(self,y,X,V):
        # phenotype
        self.y=y
        # mean covariates
        self.X=X
        # variance covariates
        self.V=V

    # Compute likelihood of data given beta, alpha
    def likelihood(self,beta,alpha):
        Vbeta = self.V.dot(beta)
        resid = self.y - self.X.dot(alpha)
        L = np.sum(Vbeta) + np.sum(np.square(resid) * np.exp(-Vbeta))
        # print('Likelihood: '+str(round(L,5)))
        return L

    # Compute MLE of alpha given beta
    def alpha_mle(self,beta):
        D_inv = np.exp(-self.V.dot(beta))
        X_t_D_inv = np.transpose(self.X) * D_inv
        alpha = np.linalg.solve(X_t_D_inv.dot(self.X), X_t_D_inv.dot(self.y))
        return alpha

    # Compute gradient with respect to beta for a given beta and alpha
    def grad_beta(self,beta, alpha):
        D_inv = np.exp(-self.V.dot(beta))
        resid_2 = np.square(self.y - self.X.dot(alpha))
        k = 1 - resid_2 * D_inv
        V_scaled = np.transpose(np.transpose(self.V) * k)
        n1t = np.ones((1, self.X.shape[0]))
        return n1t.dot(V_scaled)

    # Find an approximation to the MLE of beta given alpha
    def approx_beta_mle(self,alpha):
        resid_2=np.square(self.y-self.X.dot(alpha)).reshape((self.X.shape[0]))
        # RHS
        V_scaled=np.transpose(np.transpose(self.V)*(resid_2-1))
        n1t=np.ones((1,self.X.shape[0]))
        b=n1t.dot(V_scaled).reshape(self.V.shape[1])
        # LHS
        V_t_scaled=np.transpose(self.V)*resid_2
        A=V_t_scaled.dot(self.V)
        return np.linalg.solve(A,b)

    # Find the covariance matrix of beta
    def beta_cov(self):
        return 2 * np.linalg.inv(np.dot(self.V.T, self.V))

    # Find the covariance matrix for alpha given beta
    def alpha_cov(self,beta):
        D_inv=np.exp(-self.V.dot(beta))
        precision=np.dot(np.transpose(self.X)*D_inv,self.X)
        return np.linalg.inv(precision)

    def optimize_model(self):
        # Get initial guess for alpha
        alpha_ols = np.linalg.solve(np.dot(self.X.T, self.X), np.dot(self.X.T, self.y))
        # Get initial guess for beta
        beta_init = self.approx_beta_mle(alpha_ols)
        # Optimize
        optimized = minimize(likelihood_beta, beta_init,
                             args=(self.y, self.X, self.V),
                             method='Newton-CG',
                             jac=gradient_beta,
                             hess=H_beta_beta)
        # Get MLE
        beta_mle = optimized['x']
        alpha = self.alpha_mle(beta_mle)
        # Get parameter covariance
        optim = {}
        optim['alpha'] = alpha
        optim['beta'] = beta_mle
        optim['beta_cov'] = self.beta_cov()
        optim['beta_se'] = np.sqrt(np.diag(optim['beta_cov']))
        optim['alpha_cov'] = self.alpha_cov(beta_mle)
        optim['alpha_se'] = np.sqrt(np.diag(optim['alpha_cov']))
        optim['likelihood'] = optimized['fun']
        return optim


##### Functions to pass to opimizer ######
# Profile likelihood of hlm_model as a function of beta
def likelihood_beta(beta,*args):
    y,X,V=args
    hlm_mod=hlm_model(y,X,V)
    alpha=hlm_mod.alpha_mle(beta)
    return hlm_mod.likelihood(beta,alpha)

# Gradient of likelihood with respect to beta at the MLE of alpha
def gradient_beta(beta, *args):
    y,X,V=args
    hlm_mod=hlm_model(y,X,V)
    alpha = hlm_mod.alpha_mle(beta)
    return hlm_mod.grad_beta(beta, alpha).reshape((hlm_mod.V.shape[1]))

# Find beta component of the Hessian of the log-likelihood
def H_beta_beta(beta, *args):
    y,X,V=args
    hlm_mod=hlm_model(y,X,V)
    alpha = hlm_mod.alpha_mle(beta)
    resid_2 = np.square(hlm_mod.y - hlm_mod.X.dot(alpha))
    D_inv = np.exp(-hlm_mod.V.dot(beta))
    delta = resid_2 * D_inv
    V_t_scaled = np.transpose(hlm_mod.V) * delta
    return V_t_scaled.dot(hlm_mod.V)

####### Auxilliary functions ##########

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
    if 'fixed_mean' in args:
        # Normalise
        args.fixed_mean=args.fixed_mean[y_not_nan,:]
        fixed_mean_stds=np.std(args.fixed_mean,axis=0)
        args.fixed_mean=args.fixed_mean/fixed_mean_stds
        # Add intercept
        n_fixed_mean=args.fixed_mean.shape[1]+1
        fixed_mean=np.zeros((n,n_fixed_mean))
        fixed_mean[:,0]=np.ones(n)
        fixed_mean[:,1:n_fixed_mean]=args.fixed_mean
        fixed_mean_names=np.zeros((n_fixed_mean),dtype='S10')
        fixed_mean_names[0]='Intercept'
        fixed_mean_names[1:n_fixed_mean]=args.fixed_mean_names
    else:
        fixed_mean=np.ones((n,1))
        n_fixed_mean=1
        fixed_mean_names=np.array(['Intercept'])
    if 'fixed_variance' in args:
        # Normalise
        args.fixed_variance=args.fixed_variance[y_not_nan,:]
        fixed_var_stds=np.std(args.fixed_variance,axis=0)
        args.fixed_variance=args.fixed_variance/fixed_var_stds
        # Add scale
        n_fixed_variance=args.fixed_variance.shape[1]+1
        fixed_variance=np.zeros((n,n_fixed_variance))
        fixed_variance[:,0]=np.ones(n)
        fixed_variance[:,1:n_fixed_variance]=args.fixed_variance
        fixed_variance_names=np.zeros((n_fixed_variance),dtype='S10')
        fixed_variance_names[0]='Intercept'
        fixed_variance_names[1:n_fixed_variance]=args.fixed_variance_names
    else:
        fixed_variance=np.ones((n,1))
        n_fixed_variance=1
        fixed_variance_names=np.array(['Scale'])
    n_pars=n_fixed_mean+n_fixed_variance+1
    print(str(n_pars)+' parameters in model')
    # Rescale
    ######### Initialise output files #######
    ## Output file
    if args.append:
        write_mode='ab'
    else:
        write_mode='wb'
    outfile=open(args.outprefix+'.models.gz',write_mode)
    if not args.append:
        header='SNP_index\tfrequency\tlikelihood\tmean_effect\tmean_effect_se\tmean_effect_t\tmean_effect_pval\tvar_effect\tvar_effect_se\tvar_effect_t\tvar_effect_pval\n'
        outfile.write(header)
    ######### Fit Null Model ##########
    ## Get initial guesses for null model
    print('Fitting Null Model')
    # Assume no heteroskedasticity to start with
    # Get initial guess for fixed variance effects using linear approximation
    #beta_null=init_beta(D_inv_init,h2_init,y,fixed_mean,fixed_variance,G)
    # Optimize null model
    null=hlm_model(y,fixed_mean,fixed_variance)
    null_optim=null.optimize_model()
    #code.interact(local=locals())
    ## Record fitting of null model
    # mean effects
    beta_null=null_optim['beta']
    alpha_null=null_optim['alpha']
    null_mle_se=np.zeros((n_fixed_mean+n_fixed_variance))
    null_mle_se[0:n_fixed_mean]=np.sqrt(np.diag(null_optim['alpha_cov']))
    null_mle_se[n_fixed_mean:(n_fixed_mean+n_fixed_variance)]=np.sqrt(np.diag(null_optim['beta_cov']))
    # Get print out for fixed mean effects
    alpha_out=np.zeros((n_fixed_mean,2))
    alpha_out[:,0]=alpha_null
    alpha_out[:,1]=null_mle_se[0:n_fixed_mean]
    if not args.append:
        np.savetxt(args.outprefix + '.null_mean_effects.txt',
                   np.hstack((fixed_mean_names.reshape((n_fixed_mean, 1)), np.array(alpha_out, dtype='S20'))),
                   delimiter='\t', fmt='%s')
    # variance effects
    beta_out=np.zeros((n_fixed_variance,2))
    beta_out[0:n_fixed_variance,0]=beta_null
    beta_out[0:n_fixed_variance,1]=null_mle_se[n_fixed_mean:(n_fixed_mean+n_fixed_variance)]
    if not args.append:
        np.savetxt(args.outprefix + '.null_variance_effects.txt',
                   np.hstack((fixed_variance_names.reshape((n_fixed_variance, 1)), np.array(beta_out, dtype='S20'))),
                   delimiter='\t', fmt='%s')
    if not args.fit_mean_covariates:
        # Residual phenotype
        y=y-fixed_mean.dot(alpha_null)
        # Reformulate fixed_effects
        fixed_mean=np.ones((n,1))
        n_fixed_mean=1
    if not args.fit_variance_covariates:
        #code.interact(local=locals())
        # Residual phenotype
        lin_var_null=fixed_variance.dot(null_optim['beta'])
        D_null_sqrt=np.exp(0.5*lin_var_null)
        y=y/D_null_sqrt
        # Reformulate fixed_effects
        fixed_variance=np.ones((n,1))
        n_fixed_variance=1
        # Remake parbounds
        parbounds=[]
        for i in xrange(0,n_fixed_variance):
            parbounds.append((None,None))
        parbounds.append((0.00001,None))
    ## Loop through loci
    for loc in xrange(0,chr_length):
        print(loc)
        # Filler for output if locus doesn't pass thresholds
        additive_av_out='NaN\tNaN\tNaN\tNaN'
        variance_out='NaN\tNaN\tNaN\tNaN'
        likelihood=np.nan
        allele_frq=np.nan
        # Get test genotypes
        test_gts=genotypes[:,loc]
        # Find missingness and allele freq
        test_gt_not_na=test_gts>=0
        n_missing=float(np.sum(np.logical_not(test_gt_not_na)))
        if n_missing<n:
            missingness=100.0*(n_missing/n)
            test_gts=test_gts[test_gt_not_na]
            test_gts = test_gts.reshape((test_gts.shape[0], 1))
            allele_frq=np.mean(test_gts)/2
            if allele_frq>0.5:
                allele_frq=1-allele_frq
            if allele_frq>args.min_maf and missingness<args.max_missing:
                # Remove missing data rows
                y_l=y[test_gt_not_na]
                n_l=len(y_l)
                X_l=fixed_mean[test_gt_not_na,:]
                V_l=fixed_variance[test_gt_not_na,:]
                X_l=np.hstack((X_l,test_gts))
                V_l=np.hstack((V_l,test_gts))
                print('Fitting locus linear mean and log-linear variance model')
                # Add to parbounds
                av=hlm_model(y_l,X_l,V_l)
                av_optim=av.optimize_model()
                # Likelihood
                likelihood=-0.5*(av_optim['likelihood']+n_l*np.log(2*np.pi))
                # Mean effect of locus
                alpha_av=av_optim['alpha']
                # Approximate standard errors
                av_se=np.zeros((2))
                alpha_av_se=np.sqrt(np.diag(av_optim['alpha_cov']))
                av_se[0]=alpha_av_se[n_fixed_mean]
                beta_av_se=np.sqrt(np.diag(av_optim['beta_cov']))
                av_se[1]=beta_av_se[n_fixed_variance]
                additive_av_out=vector_out(alpha_av[n_fixed_mean],av_se[0],6)
                # Variance effect of locus
                variance_out=vector_out(av_optim['beta'][n_fixed_variance],av_se[1])
        outfile.write(str(args.start+loc) + '\t' + str(allele_frq)+'\t'+str(likelihood)+'\t'+additive_av_out+'\t'+variance_out+'\n')
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
    parser.add_argument('--h2_init',type=float,help='Initial value for variance explained by random effects (default 0.05)',
                        default=0.05)
    parser.add_argument('--phen_index',type=int,help='If phenotype file contains multiple phenotypes, which column to choose (default 1, first)',
                        default=1)
    parser.add_argument('--min_maf',type=float,help='Minimum minor allele frequency (default 0.05)',default=0.05)
    parser.add_argument('--max_missing',type=float,help='Maximum percent of missing genotype calls (default 5)',default=5)
    parser.add_argument('--missing_char',type=str,help='Missing value string in phenotype file (default NA)',default='NA')
    parser.add_argument('--fit_mean_covariates',action='store_true',default=False)
    parser.add_argument('--fit_variance_covariates',action='store_true',default=False)
    parser.add_argument('--min_obs',type=int,help='Minimum number of observations of each genotype to fit dominance/general models',default=100)
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
        args.fixed_mean=mean_covar_f.val
        args.fixed_mean_names=np.array(mean_covar_f._col,dtype='S20')
        # Match with geno_ids
        mean_ids_dict=id_dict_make(np.array(mean_covar_f.iid))
        fixed_mean_id_match=np.array([mean_ids_dict[tuple(x)] for x in geno_ids])
        args.fixed_mean=args.fixed_mean[fixed_mean_id_match,:]

    if not args.var_covar==None:
        var_covar_f=Pheno(args.var_covar,iid_if_none=geno_ids,missing=args.missing_char).read()
        args.fixed_variance=var_covar_f.val
        args.fixed_variance_names=np.array(var_covar_f._col,dtype='S20')
        # Match with geno_ids
        var_ids_dict=id_dict_make(np.array(var_covar_f.iid))
        fixed_variance_id_match=np.array([var_ids_dict[tuple(x)] for x in geno_ids])
        args.fixed_variance=args.fixed_variance[fixed_variance_id_match,:]

    learn_models_chr(args)