"""
Usage: hlmm_chr.py

This script fits heteroskedastic linear models or heteroskedastic linear mixed models to a sequence of genetic variants
contained in a .bed file. You need to specify the genotypes.bed file, which also has genotypes.bim and genotypes.fam in
the same directory, along with the start and end indices of segment you want the script to fit models to.
The script runs from start to end-1 inclusive, and the first SNP has index 0.
The script is designed to run on a chromosome segment to facilitate parallel computing on a cluster.

The phenotype file and covariate file formats are the same: FID, IID, Trait1, Trait2, ...

If you specify a random_gts.bed file with the option --random_gts, the script will model random effects for
all of the variants specified in random_gts.bed. If no --random_gts are specified, then heteroskedastic linear
models are used, without random effects.

Minimally, the script will output a file outprefix.models.gz, which contains a table of the additive
and log-linear variance effects estimated for each variant in the bed file.

If --random_gts are specified, the script will output an estimate of the variance of the random effects
in the null model in outprefix.null_h2.txt. --no_h2_estimate suppresses this output.

If covariates are also specified, it will output estimates of the covariate effects from the null model as
outprefix.null_mean_effects.txt and outprefix.null_variance_effects.txt. --no_covariate_estimates suppresses this output.
"""

from hlmm import hetlm
from hlmm import hetlmm
import argparse
import numpy as np
from pysnptools.snpreader import Bed, Pheno
from scipy.stats import chi2, zscore

####### Output functions ##########
def neglog10pval(x,df):
    return -np.log10(np.e)*chi2.logsf(x,df)

def vector_out(alpha,se,digits=4):
##Output parameter estimates along with standard errors, t-statistics, and -log10(p-values) ##
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

def id_dict_make(ids):
## Make a dictionary mapping from IDs to indices ##
    if not type(ids)==np.ndarray:
        raise(ValueError('Unsupported ID type: should be numpy nd.array'))
    id_dict={}
    for id_index in xrange(0,len(ids)):
        id_dict[tuple(ids[id_index,:])]=id_index
    return id_dict

def read_covariates(covar_file,ids_to_match,missing):
## Read a covariate file and reorder to match ids_to_match ##
    # Read covariate file
    covar_f = Pheno(covar_file, missing=missing).read()
    ids = covar_f.iid
    # Get covariate values
    n_X=covar_f._col.shape[0]+1
    X=np.ones((covar_f.val.shape[0],n_X))
    X[:, 1:n_X] = covar_f.val
    # Get covariate names
    X_names = np.zeros((n_X), dtype='S10')
    X_names[0] = 'Intercept'
    X_names[1:n_X] = np.array(covar_f._col, dtype='S20')
    # Remove NAs
    NA_rows = np.isnan(X).any(axis=1)
    n_NA_row = np.sum(NA_rows)
    if n_NA_row>0:
        print('Number of rows removed from covariate file due to missing observations: '+str(np.sum(NA_rows)))
        X = X[~NA_rows]
        ids = ids[~NA_rows]
    id_dict = id_dict_make(ids)
    # Match with pheno_ids
    ids_to_match_tuples = [tuple(x) for x in ids_to_match]
    common_ids = id_dict.viewkeys() & set(ids_to_match_tuples)
    pheno_in = np.array([(tuple(x) in common_ids) for x in ids_to_match])
    match_ids = ids_to_match[pheno_in,:]
    X_id_match = np.array([id_dict[tuple(x)] for x in match_ids])
    X = X[X_id_match, :]
    return [X,X_names,pheno_in]

######### Command line arguments #########
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mean_covar',type=str,help='Location of mean covariate file (default None)',
                        default=None)
    parser.add_argument('var_covar',type=str,help='Location of variance covariate file (default None)',
                        default=None)
    parser.add_argument('phenofile',type=str,help='Location of the phenotype file')
    parser.add_argument('outprefix',type=str,help='Location to output csv file with association statistics')
    parser.add_argument('--random_gts',type=str,help='Location of the BED file with the genotypes of the SNPs that random effects should be modelled for',default=None)
    parser.add_argument('--h2_init',type=float,help='Initial value for variance explained by random effects (default 0.05)',
                        default=0.05)
    parser.add_argument('--phen_index',type=int,help='If the phenotype file contains multiple phenotypes, which phenotype should be analysed (default 1, first)',
                        default=1)
    parser.add_argument('--missing_char',type=str,help='Missing value string in phenotype file (default NA)',default='NA')
    parser.add_argument('--no_h2_estimate',action='store_true',default=False,help='Suppress output of h2 estimate')

    args=parser.parse_args()

    ####################### Read in data #########################
    #### Read phenotype ###
    pheno = Pheno(args.phenofile, missing=args.missing_char).read()
    y = np.array(pheno.val)
    pheno_ids = np.array(pheno.iid)
    if y.ndim == 1:
        pass
    elif y.ndim == 2:
        y = y[:, args.phen_index - 1]
    else:
        raise (ValueError('Incorrect dimensions of phenotype array'))
    # Remove y NAs
    y_not_nan = np.logical_not(np.isnan(y))
    if np.sum(y_not_nan) < y.shape[0]:
        y = y[y_not_nan]
        pheno_ids = pheno_ids[y_not_nan,:]
    # Make id dictionary
    print('Number of non-missing y observations: ' + str(y.shape[0]))

    ## Get mean covariates
    if not args.mean_covar == None:
        X, X_names, pheno_in = read_covariates(args.mean_covar,pheno_ids, args.missing_char)
        n_X = X.shape[1]
        # Remove rows with missing values
        if np.sum(pheno_in) < y.shape[0]:
            y = y[pheno_in]
            pheno_ids = pheno_ids[pheno_in,:]
        # Normalise non-constant cols
        X_stds = np.std(X[:, 1:n_X], axis=0)
        X[:, 1:n_X] = zscore(X[:, 1:n_X], axis=0)
    else:
        X = np.ones((int(y.shape[0]), 1))
        n_X = 1
        X_names = np.array(['Intercept'])
    ## Get variance covariates
    if not args.var_covar == None:
        V, V_names, pheno_in = read_covariates(args.var_covar,pheno_ids, args.missing_char)
        n_V = V.shape[1]
        # Remove rows with missing values
        if np.sum(pheno_in) < y.shape[0]:
            y = y[pheno_in]
            pheno_ids = pheno_ids[pheno_in,:]
        # Normalise non-constant cols
        V_stds = np.std(V[:, 1:n_V], axis=0)
        V[:, 1:n_V] = zscore(V[:, 1:n_V], axis=0)
    else:
        V = np.ones((int(y.shape[0]), 1))
        n_V = 1
        V_names = np.array(['Intercept'])
    n_pars = n_X + n_V + 1
    print(str(n_pars) + ' parameters in model')

    # Get sample size
    n = y.shape[0]
    if n == 0:
        raise (ValueError('No non-missing observations with both phenotype and genotype data'))
    print(str(n) + ' individuals with no missing phenotype or covariate observations')
    n = float(n)

    #### Read random effect genotypes ####
    if args.random_gts is not None:
        random_gts_f = Bed(args.random_gts)
        random_gts_ids = np.array(random_gts_f.iid)
        random_gts_f = random_gts_f.read()
        # Match to phenotypes
        pheno_id_dict = id_dict_make(pheno_ids)
        G_random = random_gts_f.val
        G = np.empty((y.shape[0], G_random.shape[1]))
        G[:] = np.nan
        for i in xrange(0, random_gts_ids.shape[0]):
            if tuple(random_gts_ids[i, :]) in pheno_id_dict:
                G[pheno_id_dict[tuple(random_gts_ids[i, :])], :] = G_random[i, :]
        del G_random
        # Check for NAs
        random_isnan = np.isnan(G)
        random_gts_NAs = np.sum(random_isnan, axis=0)
        gts_with_obs = list()
        if np.sum(random_gts_NAs) > 0:
            print('Mean imputing missing genotypes in random effect design matrix')
            for i in xrange(0, G.shape[1]):
                if random_gts_NAs[i] < G.shape[0]:
                    gts_with_obs.append(i)
                    if random_gts_NAs[i] > 0:
                        gt_mean = np.mean(G[np.logical_not(random_isnan[:, i]), i])
                        G[random_isnan[:, i], i] = gt_mean
            # Keep only columns with observations
            if len(gts_with_obs) < G.shape[1]:
                G = G[:, gts_with_obs]
        G = zscore(G, axis=0)
        # Rescale random effect design matrix
        G = np.power(G.shape[1], -0.5) * G
        print(str(int(G.shape[1])) + ' loci in random effect')
    else:
        G = None

    ######### Fit  Model ##########
    ## Get initial guesses for null model
    print('Fitting Model')
    # Optimize null model
    if G is not None:
        optim= hetlmm.model(y, X, V, G).optimize_model()
        # Save h2 estimate
        if not args.no_h2_estimate:
            np.savetxt(args.outprefix + '.h2.txt',
                       np.array([optim['h2'], optim['h2_se']], dtype='S20'),
                       delimiter='\t', fmt='%s')
    else:
        optim = hetlm.model(y, X, V).optimize_model()

    ## Record fitting of model
    # Get print out for fixed mean effects
    alpha_out=np.zeros((n_X,2))
    alpha_out[:,0]=optim['alpha']
    alpha_out[:,1]=optim['alpha_se']
    # Rescale
    if n_X>1:
        for i in xrange(0,2):
            alpha_out[1:n_X,i] = alpha_out[1:n_X,i]/X_stds
    # Save
    np.savetxt(args.outprefix + '.mean_effects.txt',
               np.hstack((X_names.reshape((n_X, 1)), np.array(alpha_out, dtype='S20'))),
               delimiter='\t', fmt='%s')

    # variance effects
    beta_out=np.zeros((n_V,2))
    beta_out[0:n_V,0]=optim['beta']
    beta_out[0:n_V,1]=optim['beta_se']
    # Rescale
    if n_V>1:
        for i in xrange(0,2):
            beta_out[1:n_X,i] = beta_out[1:n_X,i]/V_stds
    # Save
    np.savetxt(args.outprefix + '.variance_effects.txt',
               np.hstack((V_names.reshape((n_V, 1)), np.array(beta_out, dtype='S20'))),
               delimiter='\t', fmt='%s')