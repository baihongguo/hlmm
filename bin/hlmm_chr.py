#!/apps/well/python/2.7.8/bin/python
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
models are used, without random effects. If you add the flag --random_gts_txt, the program assumes that the file
specified for --random_gts is a text file formatted as: FID, IID, x1, x2, ...

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
    parser.add_argument('genofile',type=str,help='Path to genotypes in BED format')
    parser.add_argument('start',type=int,help='Index of SNP in genofile from which to start computing test stats')
    parser.add_argument('end',type=int,help='Index of SNP in genofile at which to finish computing test stats')
    parser.add_argument('phenofile',type=str,help='Location of the phenotype file')
    parser.add_argument('outprefix',type=str,help='Location to output csv file with association statistics')
    parser.add_argument('--mean_covar',type=str,help='Location of mean covariate file (default None)',
                        default=None)
    parser.add_argument('--var_covar',type=str,help='Location of variance covariate file (default None)',
                        default=None)
    parser.add_argument('--fit_covariates',action='store_true',
                        help='Fit covariates for each locus. Default is to fit for null model and project out (mean) and rescale (variance)',
                        default=False)
    parser.add_argument('--random_gts',type=str,help='Location of the BED file with the genotypes of the SNPs that random effects should be modelled for',default=None)
    parser.add_argument('--random_gts_txt',action='store_true',default=False,help='Random effect design matrix supplied as text file with columns: FID, IID, x1, x2, ... Overrides assumed .bed formatting')
    parser.add_argument('--h2_init',type=float,help='Initial value for variance explained by random effects (default 0.05)',
                        default=0.05)
    parser.add_argument('--phen_index',type=int,help='If the phenotype file contains multiple phenotypes, which phenotype should be analysed (default 1, first)',
                        default=1)
    parser.add_argument('--min_maf',type=float,help='Ignore SNPs with minor allele frequency below min_maf (default 0.05)',default=0.05)
    parser.add_argument('--missing_char',type=str,help='Missing value string in phenotype file (default NA)',default='NA')
    parser.add_argument('--max_missing',type=float,help='Ignore SNPs with greater percent missing calls than max_missing (default 5)',default=5)
    parser.add_argument('--append',action='store_true',default=False,help='Append results to existing output file with given outprefix (default overwrites existing')
    parser.add_argument('--whole_chr',action='store_true',default=False,help='Fit models to all variants in .bed genofile')
    parser.add_argument('--no_covariate_estimates',action='store_true',default=False,help='Suppress output of covariate effect estimates')
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
    print('Number of non-missing phenotype observations: ' + str(y.shape[0]))

    ### Get covariates
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

    ### Read genotypes ###
    test_chr = Bed(args.genofile)
    # select subset to test
    if args.whole_chr:
        sid = test_chr.sid
        pos = test_chr.pos
        test_chr = test_chr.read()
    else:
        sid = test_chr.sid[args.start:args.end]
        pos = test_chr.pos[args.start:args.end]
        test_chr = test_chr[:, args.start:args.end].read()
    genotypes = test_chr.val
    # Get genotype matrix
    if genotypes.ndim == 1:
        chr_length = 1
        genotypes = genotypes.reshape(genotypes.shape[0], 1)
    else:
        chr_length = genotypes.shape[1]
    print('Number of test loci: ' + str(genotypes.shape[1]))
    print('Genotypes for '+str(genotypes.shape[0])+' individuals read')
    # Get sample ids
    geno_id_dict = id_dict_make(np.array(test_chr.iid))
    # Intersect with phenotype IDs
    ids_in_common = {tuple(x) for x in pheno_ids} & geno_id_dict.viewkeys()
    pheno_ids_in_common = np.array([tuple(x) in ids_in_common for x in pheno_ids])
    y = y[pheno_ids_in_common]
    pheno_ids = pheno_ids[pheno_ids_in_common,:]
    pheno_id_dict = id_dict_make(pheno_ids)
    X = X[pheno_ids_in_common,:]
    V = V[pheno_ids_in_common,:]
    geno_id_match = np.array([geno_id_dict[tuple(x)] for x in pheno_ids])
    genotypes = genotypes[geno_id_match, :]

    # Get sample size
    n = genotypes.shape[0]
    if n == 0:
        raise (ValueError('No non-missing observations with both phenotype and genotype data'))
    print(str(n) + ' individuals in genotype file with no missing phenotype or covariate observations')
    n = float(n)

    #### Read random effect genotypes ####
    if args.random_gts is not None:
        if args.random_gts_txt:
            random_gts_f = Pheno(args.random_gts)
        else:
            random_gts_f = Bed(args.random_gts)
        random_gts_ids = np.array(random_gts_f.iid)
        random_gts_f = random_gts_f.read()
        # Match to phenotypes
        G_random = random_gts_f.val
        G = np.empty((genotypes.shape[0], G_random.shape[1]))
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

    ######### Initialise output files #######
    ## Output file
    if args.append:
        write_mode='ab'
    else:
        write_mode='wb'
    outfile=open(args.outprefix+'.models.gz',write_mode)
    if not args.append:
        header='SNP\tn\tfrequency\tlikelihood\tadd\tadd_se\tadd_t\tadd_pval\tvar\tvar_se\tvar_t\tvar_pval\tav_pval\n'
        outfile.write(header)

    ######### Fit Null Model ##########
    ## Get initial guesses for null model
    print('Fitting Null Model')
    # Optimize null model
    null_optim= hetlm.model(y, X, V).optimize_model()

    ## Record fitting of null model
    # Get print out for fixed mean effects
    alpha_out=np.zeros((n_X,2))
    alpha_out[:,0]=null_optim['alpha']
    alpha_out[:,1]=null_optim['alpha_se']
    # Rescale
    if n_X>1:
        for i in xrange(0,2):
            alpha_out[1:n_X,i] = alpha_out[1:n_X,i]/X_stds
    if not args.append and not args.no_covariate_estimates and args.mean_covar is not None:
        np.savetxt(args.outprefix + '.null_mean_effects.txt',
                   np.hstack((X_names.reshape((n_X, 1)), np.array(alpha_out, dtype='S20'))),
                   delimiter='\t', fmt='%s')
    # variance effects
    beta_out=np.zeros((n_V,2))
    beta_out[0:n_V,0]=null_optim['beta']
    beta_out[0:n_V,1]=null_optim['beta_se']
    # Rescale
    if n_V>1:
        for i in xrange(0,2):
            beta_out[1:n_X,i] = beta_out[1:n_X,i]/V_stds
    if not args.append and not args.no_covariate_estimates and args.var_covar is not None:
        np.savetxt(args.outprefix + '.null_variance_effects.txt',
                   np.hstack((V_names.reshape((n_V, 1)), np.array(beta_out, dtype='S20'))),
                   delimiter='\t', fmt='%s')

    ### Project out mean covariates and rescale if not fitting for each locus
    if not args.fit_covariates:
        # Residual y
        y=y-X.dot(null_optim['alpha'])
        # Reformulate fixed_effects
        X=np.ones((int(n),1))
        n_X=1
        # Rescaled residual y
        D_null_sqrt=np.exp(0.5*V.dot(null_optim['beta']))
        y=y/D_null_sqrt
        # Reformulate fixed variance effects
        V=np.ones((int(n),1))
        n_V=1

    # Initialise h2
    if G is not None:
        if args.no_h2_estimate:
            h2_init = args.h2_init
        else:
            print('Estimating h2 in null model')
            null_optim = hetlmm.model(y, X, V, G).optimize_model(args.h2_init)
            h2_init = null_optim['h2']
            # Save null h2 estimate
            if not args.append:
                np.savetxt(args.outprefix + '.null_h2.txt',
                           np.array([null_optim['h2'], null_optim['h2_se']], dtype='S20'),
                           delimiter='\t', fmt='%s')

    ############### Loop through loci and fit AV models ######################
    print('Fitting models for specified loci')
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
            # Mean normalise genotype vector
            test_gts = test_gts - 2*allele_frq
            if allele_frq>0.5:
                allele_frq=1-allele_frq
            if allele_frq>args.min_maf:
                # Remove missing data rows
                y_l=y[test_gt_not_na]
                X_l=X[test_gt_not_na,:]
                V_l=V[test_gt_not_na,:]
                # Add test locus genotypes to mean and variance fixed effect design matrices
                X_l=np.hstack((X_l,test_gts))
                V_l=np.hstack((V_l,test_gts))
                # Record standard deviation of test gt
                print('Fitting AV model for locus '+str(loc))
                if G is not None:
                    G_l = G[test_gt_not_na, :]
                    av_optim = hetlmm.model(y_l, X_l, V_l, G_l).optimize_model(h2_init)
                    h2_init = av_optim['h2']
                else:
                    av_optim= hetlm.model(y_l, X_l, V_l).optimize_model()
                # Check convergence success
                if av_optim['success']:
                    # Likelihood
                    likelihood=av_optim['likelihood']
                    # Mean effect of locus
                    additive_av_out=vector_out(av_optim['alpha'][n_X],av_optim['alpha_se'][n_X],6)
                    # Variance effect of locus
                    variance_out=vector_out(av_optim['beta'][n_V],av_optim['beta_se'][n_V],6)
                    av_pval=neglog10pval((av_optim['alpha'][n_X]/av_optim['alpha_se'][n_X])**2+(av_optim['beta'][n_V]/av_optim['beta_se'][n_V])**2,2)
                else:
                    print('Maximisation of likelihood failed for for '+sid[loc])
        outfile.write(sid[loc] + '\t'+str(n_l)+'\t'+ str(allele_frq)+'\t'+str(likelihood)+'\t'+additive_av_out+'\t'+variance_out+'\t'+str(round(av_pval,6))+'\n')
    outfile.close()