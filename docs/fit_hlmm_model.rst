.. hlmm documentation master file, created by
   sphinx-quickstart on Wed Nov  1 10:54:40 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for fit_hlmm_model.py script
====================================


This script fits a heteroskedastic linear model (HLMs) (:class:`hetlm.model`) or heteroskedastic linear mixed model (HLMMs) (:class:`hetlmm.model`) to a given response (phenotype), mean covariates,
variance covariates, and variables to model random effects for.

The phenotype (response) file and covariate file formats are the same: plain text files with at least three columns. The first
column is family ID, and the second column is individual ID; subsequent columns are phenotype or covariate
observations. This is the same format used by GCTA and FaSTLMM.

If you specify a random_gts.bed file with the option --random_gts, the script will fit a HLMM (:class:`hetlmm.model`),
modelling random effects for the SNPs in random_gts.bed. If no --random_gts are specified, then a HLM (:class:`hetlm.model`)
is used, without random effects. If you add the flag --random_gts_txt, the program assumes that the file
specified for --random_gts is a text file formatted as: FID, IID, x1, x2, ...

If mean and/or variance covariates are specified, the script will output two files: outprefix.mean_effects.txt, containing the estimated mean
effects and their standard errors; and outprefix.variance_effects.txt, containing the estimated log-linear
variance effects and their standard errors.

If --random_gts are specified, the script will output an estimate of the variance of the random effects
in outprefix.h2.txt. --no_h2_estimate suppresses this output.

**Arguments**

Required positional arguments:

**phenofile**
   Location of the phenotype (response) file with format: FID, IID, y1, y2, ...

**outprefix**
   Location to output csv file with association statistics

Options:

--mean_covar
   Location of mean covariate file (default no mean covariates)

--var_covar
   Locaiton of variance covariate file (default no variance covariates)

--random_gts
   Location of the BED file with the genotypes of the SNPs that random effects should be modelled for. If
   random_gts are provided, HLMMs (:class:`hetlmm.model`) are fit, rather than HLMs (:class:`hetlm.model`).

--random_gts_txt
   If this flag is specified, the program assumes the file specified in for --random_gts is formatted as a text file
   with at least three columns: FID, IID, x1, x2, ...

--h2_init
   Initial value for variance explained by random effects (default 0.05)

--phen_index
   If the phenotype file contains multiple phenotypes, specify the phenotype to analyse. Default is first phenotype in file.
   Index counts starting from 1, the first phenotype in the phenotye file.

--missing_char
   Missing value string in phenotype file (default NA)

--no_h2_estimate
    Suppress output of h2 estimate


**Example Usage**

We recommend working through the tutorial (:doc:`tutorial`) to learn how to use hlmm_chr.py. The usage of this script is similar, if
a little simpler.