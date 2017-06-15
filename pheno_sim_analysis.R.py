lhm = imp.load_source('lhm', '/well/donnelly/glmm/hlmm/linear_heteroskedastic_model.py')
import h5py, argparse
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('genofile',type=str,help='Location of the .hdf5 file with genotypes as dataset')
parser.add_argument('phenofile',type=str,help='Location of the .hdf5 file with phenotypes as dataset')
parser.add_argument('outprefix',type=str,help='Location to output csv file with test statistics')

args=parser.parse_args()

    # Get test loci
test_chr=h5py.File(args.genofile,'r')
test_gts=test_chr['genotypes']
# select subset to test
args.genotypes=np.transpose(np.array(test_gts[0,:]))
# Get sample ids
geno_ids=np.array(test_chr['sample_id'])


phenofile=h5py.File(args.phenofile,'r')
args.phenotype=np.array(phenofile['phenotypes'])

# Match IDs with geno IDs
pheno_ids=np.array(phenofile['sample_id'])
pheno_id_dict=id_dict_make(pheno_ids)
pheno_id_match=np.array([pheno_id_dict[x] for x in geno_ids])
args.phenotype=args.phenotype[pheno_id_match]

learn_models_chr(args)