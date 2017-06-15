#!/apps/well/python/2.7.8/bin/python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import h5py, argparse, code

def likelihood(y,X,V,beta,alpha):
    Vbeta=V.dot(beta)
    resid=y-X.dot(alpha)
    L=np.sum(Vbeta)+np.sum(np.square(resid)*np.exp(-Vbeta))
    #print('Likelihood: '+str(round(L,5)))
    return L

def alpha_mle(y,X,V,beta):
    D_inv=np.exp(-V.dot(beta))
    X_t_D_inv=np.transpose(X)*D_inv
    alpha=np.linalg.solve(X_t_D_inv.dot(X),X_t_D_inv.dot(y))
    return alpha

def grad_beta(y,X,V,beta,alpha):
    D_inv=np.exp(-V.dot(beta))
    resid_2=np.square(y-X.dot(alpha))
    k=1-resid_2*D_inv
    V_scaled=np.transpose(np.transpose(V)*k)
    n1t=np.ones((1,X.shape[0]))
    return n1t.dot(V_scaled)

def approx_beta_mle(y,X,V,alpha):
    resid_2=np.square(y-X.dot(alpha)).reshape((X.shape[0]))
    # RHS
    V_scaled=np.transpose(np.transpose(V)*(resid_2-1))
    n1t=np.ones((1,X.shape[0]))
    b=n1t.dot(V_scaled).reshape(V.shape[1])
    # LHS
    V_t_scaled=np.transpose(V)*resid_2
    A=V_t_scaled.dot(V)
    return np.linalg.solve(A,b)

def likelihood_beta(beta,*args):
    y, X , V = args
    alpha=alpha_mle(y,X,V,beta)
    return likelihood(y,X,V,beta,alpha)

def gradient_beta(beta,*args):
    y, X , V = args
    alpha=alpha_mle(y,X,V,beta)
    return grad_beta(y,X,V,beta,alpha).reshape((V.shape[1]))


def H_beta_beta(beta,*args):
    y, X , V = args
    alpha=alpha_mle(y,X,V,beta)
    resid_2=np.square(y-X.dot(alpha))
    D_inv=np.exp(-V.dot(beta))
    delta=resid_2*D_inv
    V_t_scaled=np.transpose(V)*delta
    return V_t_scaled.dot(V)

def alpha_cov(X,V,beta):
    D_inv=np.exp(-V.dot(beta))
    precision=np.dot(np.transpose(X)*D_inv,X)
    return np.linalg.inv(precision)

def beta_cov(V):
    return 2*np.linalg.inv(np.dot(V.T,V))

def optimize_model(y,X,V):
    # Get initial guess for alpha
    alpha_ols=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,y))
    # Get initial guess for beta
    beta_init=approx_beta_mle(y,X,V,alpha_ols)
    #Optimize
    optimized=minimize(likelihood_beta,beta_init,
                       args=(y,X,V),
                       method='Newton-CG',
                       jac=gradient_beta,
                       hess=H_beta_beta)
    # Get MLE
    beta_mle=optimized['x']
    alpha=alpha_mle(y,X,V,beta_mle)
    # Get parameter covariance
    optim={}
    optim['alpha']=alpha
    optim['beta']=beta_mle
    optim['beta_cov']=beta_cov(V)
    optim['beta_se']=np.sqrt(np.diag(optim['beta_cov']))
    optim['alpha_cov']=alpha_cov(X,V,beta_mle)
    optim['alpha_se']=np.sqrt(np.diag(optim['alpha_cov']))
    optim['likelihood']=optimized['fun']
    return optim

def neglog10pval(x,df):
    return -np.log10(np.e)*chi2.logsf(x,df)

def vector_out(alpha,se,digits=4):
    # Create output strings
    if len(alpha.shape)==0:
        alpha_print=str(round(alpha,digits))+'\t'+str(round(se,digits))
    else:
        alpha_print=''
        for i in xrange(0,len(alpha)-1):
            alpha_print+=str(round(alpha[i],digits))+'\t'+str(round(se[i],digits))+'\t'
        i+=1
        alpha_print+=str(round(alpha[i],digits))+'\t'+str(round(se[i],digits))
    return alpha_print

def learn_models_chr(args):
    # Get genotype matrix
    genotypes=args.genotypes
    if genotypes.ndim==1:
        chr_length=1
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
    #code.interact(local=locals())
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
        header='SNP_index\tfrequency\tmean_llr\tmean_effect\tmean_effect_se\tvar_llr\tmean_effect_av\tmean_effect_av_se\tvar_effect\tvar_effect_se'
        if args.dom or args.gvar:
            if args.dom:
                header+='\tdom_llr\tmean_effect_adv\tmean_effect_adv_se\tvar_effect_adv\tvar_effect_adv_se\tdom_effect\tdom_effect_se'
            if args.gvar:
                header+='\tgvar_llr\tmean_effect_advg\tmean_effect_advg_se\tvar_effect_advg\tvar_effect_advg_se\tdom_effect_advg\tdom_effect_advg_se\tgvar_effect\tgvar_effect_se\n'
            else:
                header+='\n'
        else:
            header+='\n'
        outfile.write(header)
    ######### Fit Null Model ##########
    ## Get initial guesses for null model
    print('Fitting Null Model')
    # Assume no heteroskedasticity to start with
    # Get initial guess for fixed variance effects using linear approximation
    #beta_null=init_beta(D_inv_init,h2_init,y,fixed_mean,fixed_variance,G)
    # Optimize null model
    null=optimize_model(y,fixed_mean,fixed_variance)
    ## Record fitting of null model
    # mean effects
    beta_null=null['beta']
    alpha_null=alpha_mle(y, fixed_mean, fixed_variance, beta_null)
    null_mle_se=np.zeros((n_fixed_mean+n_fixed_variance))
    null_mle_se[0:n_fixed_mean]=np.sqrt(np.diag(alpha_cov(fixed_mean,fixed_variance,beta_null)))
    null_mle_se[n_fixed_mean:(n_fixed_mean+n_fixed_variance)]=np.sqrt(np.diag(beta_cov(fixed_variance)))
    # Get print out for fixed mean effects
    alpha_out=np.zeros((n_fixed_mean,2))
    alpha_out[:,0]=alpha_null
    alpha_out[:,1]=null_mle_se[0:n_fixed_mean]
    if not args.append:
        np.savetxt(args.outprefix+'.null_mean_effects.txt',
                                  np.hstack((fixed_mean_names.reshape((n_fixed_mean,1)),alpha_out)),
                                  delimiter='\t',fmt='%s')
    # variance effects
    beta_out=np.zeros((n_fixed_variance,2))
    beta_out[0:n_fixed_variance,0]=beta_null
    beta_out[0:n_fixed_variance,1]=null_mle_se[n_fixed_mean:(n_fixed_mean+n_fixed_variance)]
    if not args.append:
        np.savetxt(args.outprefix+'.null_variance_effects.txt',
                                  np.hstack((fixed_variance_names.reshape((n_fixed_variance,1)),beta_out)),
                                  delimiter='\t',fmt='%s')
    if not args.fit_mean_covariates:
        # Residual phenotype
        y=y-fixed_mean.dot(alpha_null)
        # Reformulate fixed_effects
        fixed_mean=np.ones((n,1))
        n_fixed_mean=1
    if not args.fit_variance_covariates:
        #code.interact(local=locals())
        # Residual phenotype
        lin_var_null=fixed_variance.dot(null['beta'])
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
        additive_out='NaN\tNaN'
        additive_av_out='NaN\tNaN'
        variance_out='NaN\tNaN'
        n_models=3
        if args.dom:
            n_models+=1
            avd_out='NaN\tNaN\tNaN\tNaN\tNaN\tNaN'
        if args.gvar:
            n_models+=1
            avdg_out='NaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN'
        likelihoods=np.empty((n_models))
        likelihoods[:]=np.nan
        llrs=np.empty((n_models-1))
        llrs[:]=np.nan
        allele_frq=np.nan
        # Get test genotypes
        test_gts=genotypes[:,loc]
        # Find missingness and allele freq
        test_gt_not_na=test_gts>=0
        n_missing=float(np.sum(np.logical_not(test_gt_not_na)))
        if n_missing<n:
            missingness=100.0*(n_missing/n)
            test_gts=test_gts[test_gt_not_na]
            allele_frq=np.mean(test_gts)/2
            if allele_frq>0.5:
                allele_frq=1-allele_frq
            if allele_frq>args.min_maf and missingness<args.max_missing:
                # Remove missing data rows
                y_l=y[test_gt_not_na]
                n_l=len(y_l)
                X_l=fixed_mean[test_gt_not_na,:]
                V_l=fixed_variance[test_gt_not_na,:]
                ## Fit Null model ##
                print('Fitting locus null model')
                null_l=optimize_model(y_l,X_l,V_l)
                likelihoods[0]=-0.5*(null_l['likelihood']+n_l*np.log(2*np.pi))
                ## Fit linear mean model ##
                print('Fitting locus linear model')
                test_gts=np.array(test_gts).reshape((n_l,1))
                X_l=np.hstack((X_l,test_gts))
                # Calculate initial parameters
                additive=optimize_model(y_l,X_l,V_l)
                likelihoods[1]=-0.5*(additive['likelihood']+n_l*np.log(2*np.pi))
                beta_additive=additive['beta']
                alpha_additive=alpha_mle(y_l, X_l, V_l, beta_additive)
                # Estimate standard errors
                additive_se=np.sqrt(np.diag(alpha_cov(X_l,V_l,beta_additive)))
                additive_out=vector_out(alpha_additive[n_fixed_mean],additive_se[n_fixed_mean],6)
                ## Fit linear mean and variance model ##
                print('Fitting locus linear mean and log-linear variance model')
                V_l=np.hstack((V_l,test_gts))
                # Add to parbounds
                av=optimize_model(y_l,X_l,V_l)
                # Likelihood
                likelihoods[2]=-0.5*(av['likelihood']+n_l*np.log(2*np.pi))
                # Mean effect of locus
                alpha_av=alpha_mle(y_l, X_l, V_l, av['beta'])
                # Approximate standard errors
                av_se=np.zeros((2))
                alpha_av_se=np.sqrt(np.diag(alpha_cov(X_l,V_l,av['beta'])))
                av_se[0]=alpha_av_se[n_fixed_mean]
                beta_av_se=np.sqrt(np.diag(beta_cov(V_l)))
                av_se[1]=beta_av_se[n_fixed_variance]
                additive_av_out=vector_out(alpha_av[n_fixed_mean],av_se[0],6)
                # Variance effect of locus
                variance_out=vector_out(av['beta'][n_fixed_variance],av_se[1])
                ## Fit model with dominance
                min_obs=np.min(np.array([np.sum(test_gts==x) for x in xrange(0,3)]))
                if min_obs>args.min_obs:
                    if args.dom:
                        print('Fitting locus general mean and log-linear variance model')
                        X_l=np.hstack((X_l,np.square(test_gts)))
                        #code.interact(local=locals())
                        # Add to parbounds
                        avd=optimize_model(y_l,X_l,V_l)
                        # Likelihood
                        likelihoods[3]=-0.5*(avd['likelihood']+n_l*np.log(2*np.pi))
                        # Approximate standard errors
                        avd_mles=np.zeros((3))
                        avd_ses=np.zeros((3))
                        # Mean effect of locus
                        avd_alpha_mle=alpha_mle(y_l, X_l, V_l, avd['beta'])
                        avd_alpha_cov=np.sqrt(np.diag(alpha_cov(X_l,V_l,avd['beta'])))
                        avd_mles[0]=avd_alpha_mle[n_fixed_mean]
                        avd_ses[0]=avd_alpha_cov[n_fixed_mean]
                        # LLV effect
                        avd_mles[1]=avd['beta'][n_fixed_variance]
                        avd_ses[1]=np.sqrt(np.diag(beta_cov(V_l)))[n_fixed_variance]
                        # Dominance effect
                        avd_mles[2]=avd_alpha_mle[n_fixed_mean+1]
                        avd_ses[2]=avd_alpha_cov[n_fixed_mean+1]
                        # Output
                        avd_out=vector_out(avd_mles,avd_ses,6)
                    if args.gvar:
                        print('Fitting locus general model')
                        V_l=np.hstack((V_l,np.square(test_gts)))
                        # Add to parbounds
                        avdg=optimize_model(y_l,X_l,V_l)
                        # Likelihood
                        likelihoods[4]=-0.5*(avdg['likelihood']+n_l*np.log(2*np.pi))
                        # Approximate standard errors
                        avdg_mles=np.zeros((4))
                        avdg_ses=np.zeros((4))
                        # Mean effect of locus
                        avdg_alpha_mle=alpha_mle(y_l, X_l, V_l, avdg['beta'])
                        avdg_alpha_cov=np.sqrt(np.diag(alpha_cov(X_l,V_l,avdg['beta'])))
                        avdg_mles[0]=avdg_alpha_mle[n_fixed_mean]
                        avdg_ses[0]=avdg_alpha_cov[n_fixed_mean]
                        # LLV effect
                        avdg_mles[1]=avdg['beta'][n_fixed_variance]
                        avdg_beta_cov=np.sqrt(np.diag(beta_cov(V_l)))
                        avdg_ses[1]=avdg_beta_cov[n_fixed_variance]
                        # Dominance effect
                        avdg_mles[2]=avdg_alpha_mle[n_fixed_mean+1]
                        avdg_ses[2]=avdg_alpha_cov[n_fixed_mean+1]
                        # General varianc effect
                        avdg_mles[3]=avdg['beta'][n_fixed_variance+1]
                        avdg_ses[3]=avdg_beta_cov[n_fixed_variance+1]
                        # Output
                        avdg_out=vector_out(avdg_mles,avdg_ses,6)
                ## Write output ##
                # Chi-square statistics
                for i in xrange(0,n_models-1):
                    llrs[i]=2*(likelihoods[i+1]-likelihoods[i])
            # General association
        outline=str(args.start+loc) + '\t' + str(allele_frq)+'\t'+str(llrs[0])+'\t'+additive_out+'\t'+str(llrs[1])+'\t'+additive_av_out+'\t'+variance_out
        if args.dom or args.gvar:
            if args.dom:
                outline+='\t'+str(llrs[2])+'\t'+avd_out
            if args.gvar:
                outline+='\t'+str(llrs[3])+'\t'+avdg_out+'\n'
            else:
                outline+='\n'
        else:
            outline+='\n'
        outfile.write(outline)
    outfile.close()
    return

def id_dict_make(ids):
    id_dict={}
    for id_index in xrange(0,len(ids)):
        id_dict[ids[id_index]]=id_index
    return id_dict


if __name__ == "__main__":
    ######## Parse Arguments #########
    parser=argparse.ArgumentParser()
    parser.add_argument('genofile',type=str,help='Location of the .hdf5 file with genotypes as dataset')
    parser.add_argument('start',type=int,help='Index of locus in genofile from which to start computing test stats')
    parser.add_argument('end',type=int,help='Index of locus in genofile at which to finish computing test stats')
    parser.add_argument('phenofile',type=str,help='Location of the .hdf5 file with phenotypes as dataset')
    parser.add_argument('outprefix',type=str,help='Location to output csv file with test statistics')
    parser.add_argument('--mean_covar',type=str,help='Location of .hdf5 file with matrix of fixed mean effect variables',
                        default=None)
    parser.add_argument('--variance_covar',type=str,help='Locaiton of .hdf5 file with matrix of fixed variance effects',
                        default=None)
    parser.add_argument('--phen_index',type=int,help='If phenotype file contains multiple phenotypes, which row to choose (default 0)',
                        default=0)
    parser.add_argument('--min_maf',type=float,help='Minimum minor allele frequency',default=0.05)
    parser.add_argument('--max_missing',type=float,help='Maximum percent of missing genotype calls',default=1)
    parser.add_argument('--fit_mean_covariates',action='store_true',default=False)
    parser.add_argument('--fit_variance_covariates',action='store_true',default=False)
    parser.add_argument('--dom',action='store_true',default=False)
    parser.add_argument('--gvar',action='store_true',default=False)
    parser.add_argument('--min_obs',type=int,help='Minimum number of observations of each genotype to fit dominance/general models',default=100)
    parser.add_argument('--append',action='store_true',default=False,help='Append results to existing output file')

    args=parser.parse_args()

        # Get test loci
    test_chr=h5py.File(args.genofile,'r')
    test_gts=test_chr['genotypes']
    # select subset to test
    if args.end>test_gts.shape[0]:
        args.end=test_gts.shape[0]
    args.genotypes=np.transpose(np.array(test_gts[args.start:args.end,:]))
    print('Number of test loci: '+str(args.genotypes.shape[1]))
    # Get sample ids
    geno_ids=np.array(test_chr['sample_id'])


    phenofile=h5py.File(args.phenofile,'r')
    args.phenotype=np.array(phenofile['phenotypes'])
    if args.phenotype.ndim==1:
        pheno_noNA=args.phenotype[np.logical_not(np.isnan(args.phenotype))]
        args.phenotype=args.phenotype/pheno_noNA.std()
    elif args.phenotype.ndim==2:
        args.phenotype=args.phenotype[:,args.phen_index]
        pheno_noNA=args.phenotype[np.logical_not(np.isnan(args.phenotype))]
        args.phenotype=args.phenotype/pheno_noNA.std()
    else:
        raise(ValueError('Incorrect dimensions of phenotype array'))
    print('Number of phenotype observations: '+str(args.phenotype.shape[0]))
    # Match IDs with geno IDs
    pheno_ids=np.array(phenofile['sample_id'])
    pheno_id_dict=id_dict_make(pheno_ids)
    pheno_id_match=np.array([pheno_id_dict[x] for x in geno_ids])
    args.phenotype=args.phenotype[pheno_id_match]

    ## Get covariates
    if not args.mean_covar==None:
        mean_covar_f=h5py.File(args.mean_covar,'r')
        args.fixed_mean=np.transpose(mean_covar_f['covariates'])
        args.fixed_mean_names=np.array(mean_covar_f['names'],dtype='S20')
        # Match with geno_ids
        mean_ids_dict=id_dict_make(np.array(mean_covar_f['sample_id']))
        fixed_mean_id_match=np.array([mean_ids_dict[x] for x in geno_ids])
        args.fixed_mean=args.fixed_mean[fixed_mean_id_match,:]

    if not args.variance_covar==None:
        variance_covar_f=h5py.File(args.variance_covar,'r')
        args.fixed_variance=np.transpose(variance_covar_f['covariates'])
        args.fixed_variance_names=np.array(variance_covar_f['names'],dtype='S20')
        # Match with geno_ids
        var_ids_dict=id_dict_make(np.array(variance_covar_f['sample_id']))
        fixed_variance_id_match=np.array([var_ids_dict[x] for x in geno_ids])
        args.fixed_variance=args.fixed_variance[fixed_variance_id_match,:]

    learn_models_chr(args)