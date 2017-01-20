import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import chi2
from scipy import linalg
import argparse
from pysnptools.snpreader import Bed

def id_dict_make(ids):
    id_dict={}
    for id_index in xrange(0,len(ids)):
        id_dict[ids[id_index]]=id_index
    return id_dict

if __name__ == "__main__":
    ######## Parse Arguments #########
    parser=argparse.ArgumentParser()
    parser.add_argument('genofile',type=str,help='Path to genotypes in BED format')
    parser.add_argument('start',type=int,help='Index of SNP in genofile from which to start computing test stats')
    parser.add_argument('end',type=int,help='Index of SNP in genofile at which to finish computing test stats')
    parser.add_argument('phenofile',type=str,help='Location of the phenotype file in PLINK format')
    parser.add_argument('random_gts',type=str,help='Location of the BED file with the genotypes of the random effect')
    parser.add_argument('outprefix',type=str,help='Location to output csv file')
    parser.add_argument('--mean_covar',type=str,help='Location of mean covariate file in PLINK format',
                        default=None)
    parser.add_argument('--variance_covar',type=str,help='Locaiton of variance covariate file in PLINK format',
                        default=None)
    parser.add_argument('--h2_init',type=float,help='Initial value for variance explained by random effects (default 0.05)',
                        default=0.05)
    parser.add_argument('--phen_index',type=int,help='If phenotype file contains multiple phenotypes, which row to choose (default 0)',
                        default=0)
    parser.add_argument('--min_maf',type=float,help='Minimum minor allele frequency',default=0.05)
    parser.add_argument('--max_missing',type=float,help='Maximum percent of missing genotype calls',default=5)

    args=parser.parse_args()

    # Get test loci
    test_chr=Bed(args.genofile)
    # select subset to test
    test_gts=test_chr[:,args.start:args.end].val
    print('Number of test loci: '+str(args.genotypes.shape[1]))
    # Get sample ids
    geno_ids=test_chr.iid
    # Get phenotype file
    phenofile=h5py.File(args.phenofile,'r')
    args.phenotype=np.array(phenofile['phenotypes'])
    if args.phenotype.ndim==1:
        pheno_noNA=args.phenotype[np.logical_not(np.isnan(args.phenotype))]
        args.phenotype=args.phenotype/pheno_noNA.std()
    elif args.phenotype.ndim==2:
        args.phenotype=args.phenotype[args.phen_index,:]
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

    # Get random effect loci
    random_gts_f=Bed(args.random_gts,'r')
    args.selected_genotypes=np.transpose(random_gts_f['genotypes'])
    # Match with geno IDs
    random_ids_dict=id_dict_make(np.array(random_gts_f['sample_id']))
    random_ids_match=np.array([random_ids_dict[x] for x in geno_ids])
    args.selected_genotypes=args.selected_genotypes[random_ids_match,:]

    #
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

    #code.interact(local=locals())
    learn_models_chr(args)