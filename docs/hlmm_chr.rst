.. hlmm documentation master file, created by
   sphinx-quickstart on Wed Nov  1 10:54:40 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for hlmm_chr.py script
====================================


This script fits heteroskedastic linear models (HLMs) (:class:`hetlm.model`) or heteroskedastic linear mixed models (HLMMs) (:class:`hetlmm.model`) to a sequence of genetic variants
contained in a .bed file. You need to specify the genotypes.bed file, which also has genotypes.bim and genotypes.fam in
the same directory, along with the start and end indices of segment you want the script to fit models to.

The script runs from start to end-1 inclusive, and the first SNP has index 0.
The script is designed to run on a chromosome segment to facilitate parallel computing on a cluster.

The phenotype file and covariate file formats are the same: plain text files with at least three columns. The first
column is family ID, and the second column is individual ID; subsequent columns are phenotype or covariate
observations. This is the same format used by GCTA and FaSTLMM.

If you specify a random_gts.bed file with the option --random_gts, the script will fit HLMs (:class:`hetlm.model`),
modelling random effects for the SNPs in random_gts.bed. If no --random_gts are specified, then HLMMs (:class:`hetlmm.model`)
are used, without random effects.

Minimally, the script will output a file outprefix.models.gz, which contains a table of the additive
and log-linear variance effects estimated for each variant specified.

If --random_gts are specified, the script will output an estimate of the variance of the random effects
in the null model in outprefix.null_h2.txt. --no_h2_estimate suppresses this output.

If covariates are also specified, it will output estimates of the covariate effects from the null model as
outprefix.null_mean_effects.txt and outprefix.null_variance_effects.txt. --no_covariate_estimates suppresses this output.

***Arguments***

Required positional arguments:

**genofile**
   Path to genotypes in BED format

**start**
   Index of SNP in genofile from which to start computing test stats

**end**
   Index of SNP in genofile at which to finish computing test stats

**phenofile**
   Location of the y file in PLINK format

**outprefix**
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
   random_gts are provided, HLMMs (:class:`hetlmm.model`) are fit, rather than HLMs (:class:`hetlm.model`).

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


**Example Usage**

We recommend working through the tutorial (:doc:`tutorial`) to learn how to use hlmm_chr.py. We provide some additional
examples of usage of the script here.

Minimal usage for fitting HLMs (:class:`hetlm.model`):

   ``python hlmm_chr.py genotypes.bed 0 500 phenotype.fam phenotype``

This will fit heteroskedastic linear models to SNPs 0 to 499 in genotypes.bed using the first phenotype in phenotype.fam. It will output
the results of fitting the models to phenotype.models.gz. See :doc:`tutorial` for a description of the columns
of phenotypes.models.gz.

Minimal usage for HLMMs (:class:`hetlmm.model`):

   ``python hlmm_chr.py genotypes.bed 0 500 phenotype.fam phenotype --random_gts random.bed``

This will fit heteroskedastic linear mixed models to SNPs 0 to 499 in genotypes.bed using the first phenotype in phenotype.fam. It will output
the results of fitting the models to phenotype.models.gz. It will also output the estimate of h2, the variance
of the random effects, to phenotype.null_h2.txt, unless --no_h2_estimate is added to the command.

Fitting covariates:

   ``python hlmm_chr.py genotypes.bed 0 500 phenotype.fam phenotype --mean_covar m_covariates.fam --var_covar v_covariates.fam``

Before fitting locus specific models, the script will first fit a null model including the mean covariates in m_covariates.fam and the variance covariates in v_covariates.fam.
The script will output the null model estimates of the mean covariates in phenotype.null_mean_effects.txt and
null model estimates of the variance covariates in phenotypes.null_variance_effects.txt, unless --no_covariate_estimates is added to the command.
Unless --fit_covariates is added to the command, phenotype is adjusted based on the null model estimates of the mean
covariate effects and variance covariate effects. The adjusted phenotype is used to fit locus specific models without
fitting the mean and variance covariates.