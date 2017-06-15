import h5py, argparse, imp
import numpy as np
lhm = imp.load_source('lhm', '/well/donnelly/glmm/hlmm/linear_heteroskedastic_model.py')


parser=argparse.ArgumentParser()
parser.add_argument('genofile',type=str,help='Location of the .hdf5 file with genotypes as dataset')
parser.add_argument('phenofile',type=str,help='Location of the .hdf5 file with phenotypes as dataset')
parser.add_argument('outprefix',type=str,help='Location to output csv file with test statistics')
parser.add_argument('--mean_covar', type=str, help='Location of .hdf5 file with matrix of fixed mean effect variables',
                    default=None)
parser.add_argument('--variance_covar', type=str, help='Locaiton of .hdf5 file with matrix of fixed variance effects',
                    default=None)
parser.add_argument('--phen_index', type=int,
                    help='If phenotype file contains multiple phenotypes, which row to choose (default 0)',
                    default=0)
parser.add_argument('--min_maf', type=float, help='Minimum minor allele frequency', default=0.05)
parser.add_argument('--max_missing', type=float, help='Maximum percent of missing genotype calls', default=1)
parser.add_argument('--fit_mean_covariates', action='store_true', default=False)
parser.add_argument('--fit_variance_covariates', action='store_true', default=False)
parser.add_argument('--dom', action='store_true', default=False)
parser.add_argument('--gvar', action='store_true', default=False)
parser.add_argument('--min_obs', type=int,
                    help='Minimum number of observations of each genotype to fit dominance/general models', default=100)

args=parser.parse_args()

    # Get test loci
test_chr=h5py.File(args.genofile,'r')
test_gts=test_chr['genotypes']
# select subset to test
args.genotypes=np.transpose(np.array(test_gts))

# Get sample ids
geno_ids=np.array(test_chr['sample_id'])
args.append=True

phenofile=h5py.File(args.phenofile,'r')
phenotypes=np.array(phenofile['phenotypes']).T

print(phenotypes.shape)

for p in xrange(0,phenotypes.shape[1]):
    if p==0:
        args.append=False
    else:
        args.append=True
    args.phenotype=phenotypes[:,p]
    lhm.learn_models_chr(args)