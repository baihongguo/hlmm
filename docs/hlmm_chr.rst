.. hlmm documentation master file, created by
   sphinx-quickstart on Wed Nov  1 10:54:40 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for hlmm_chr.py script
====================================


This script fits heteroskedastic linear models or heteroskedastic linear mixed models to a sequence of genetic variants
contained in a .bed file. You need to specify the genotypes.bed file, which also has genotypes.bim and genotypes.fam in
the same directory, along with the start and end indices of segment you want the script to fit models to.

The script runs from start to end-1 inclusive, and the first SNP has index 0.
The script is designed to run on a chromosome segment to facilitate parallel computing on a cluster.

The phenotype file and covariate file formats are the same: plain text files with at least three columns. The first
column is family ID, and the second column is individual ID; subsequent columns are phenotype or covariate
observations. This is the same format used by GCTA and FaSTLMM.

If you specify a random_gts.bed file with the option --random_gts, the script will model random effects for
all of the variants specified in random_gts.bed. If no --random_gts are specified, then heteroskedastic linear
models are used, without random effects.

Minimally, the script will output a file outprefix.models.gz, which contains a table of the additive
and log-linear variance effects estimated for each variant specified.

If --random_gts are specified, the script will output an estimate of the variance of the random effects
in the null model in outprefix.null_h2.txt. --no_h2_estimate suppresses this output.

If covariates are also specified, it will output estimates of the covariate effects from the null model as
outprefix.null_mean_effects.txt and outprefix.null_variance_effects.txt. --no_covariate_estimates suppresses this output.

Required positional arguments:

genofile
   Path to genotypes in BED format

start
   Index of SNP in genofile from which to start computing test stats

end
   Index of SNP in genofile at which to finish computing test stats

phenofile
   Location of the y file in PLINK format

outprefix
   Location to output csv file with association statistics

Options:

--mean_covar
   Location of mean covariate file (default no mean covariates)

--var_covar
   Locaiton of variance covariate file (default no variance covariates)

--fit_covariates
   Fit covariates for each locus. Default is to fit covariates for the null model and project out (mean) and rescale (variance)'

--random_gts
   Location of the BED file with the genotypes of the SNPs that random effects should be modelled for. If
   random_gts are provided, heteroskedastic linear mixed models are fit, rather than heteroskedastic linear models.

--h2_init
   Initial value for variance explained by random effects (default 0.05)

--phen_index
   If the phenotype file contains multiple phenotypes, specify the phenotype to analyse. Default is first phenotype in file.
   Index counts starting from 1, the first phenotype in the phenotye file.

--min_maf
   Ignore SNPs with minor allele frequency below min_maf (default 5%)

--missing_char
   Missing value string in phenotype file (default NA)

--max_missing
   Ignore SNPs with greater % missing calls than max_missing (default 5%)

--append
   Append results to existing output file with given outprefix (default to open new file and overwrite existing file with same name)

--whole_chr
   Fit models to all variants in .bed genofile. Overrides default to model SNPs with indices from start to end-1.

--no_covariate_estimates
   Suppress output of covariate effect estimates

--no_h2_estimate
    Suppress output of h2 estimate
