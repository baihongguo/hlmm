#!/apps/well/python/2.7.8/bin/python

__author__ = 'ay'
import numpy as np
from numpy import random as rand
import argparse, h5py, math, code

def id_dict_make(ids):
    id_dict={}
    for id_index in xrange(0,len(ids)):
        id_dict[ids[id_index]]=id_index
    return id_dict

parser=argparse.ArgumentParser(description='Simulate Phenotypes with Particular Parameters.h')

parser.add_argument('genofiles',type=str,help='Text file with location of genotypes for chr i on line i')
parser.add_argument('phenofile',type=str,help='Location of hdf5 file to output phenotypes to')
parser.add_argument('nphen',type=int,help='Number of phenotypes to simulate')
parser.add_argument('nc',type=int,help='Number of Causal Variants in Model')
parser.add_argument('nd',type=int,help='Number of causal variants with dominance effects')
parser.add_argument('n_interact',type=int,help='Number of pairs of causal variants that interact')
parser.add_argument('h2',type=float,help='Narrow Sense Heritability')
parser.add_argument('hd',type=float,help='Proportion of Variance from Dominance Effects')
parser.add_argument('h2_epi',type=float,help='Proportion of Variance from Interactions')
parser.add_argument('--save_causal_gts',action='store_true',default=False)
#parser.add_argument('--seed',type=int,help='Set seed for random sampling of genetic effects',
#                    default=64)
parser.add_argument('--min_maf',type=float,help='Minimum minor allele frequency of causal loci',
                    default=0.05)
parser.add_argument('--max_missing',type=float,help='Maximum percentage missing calls for causal locus',
                    default=1)
parser.add_argument('--normal_effects',action='store_true',default=False,
                    help='Normally distributed effect size distribution. Default is equal effect size but random sign.')
parser.add_argument('--variance_effect',type=float,help='Size of variance effect for loci with additive effect',default=0.0)
args=parser.parse_args()
genofiles=args.genofiles
phenout=args.phenofile
nphen=args.nphen
nc=args.nc
nd=args.nd
n_interact=args.n_interact
h2=args.h2
hd=args.hd
h2_epi=args.h2_epi

# Set random seed
#rand.seed(args.seed)

def dom_convert(g,f):
    g=int(g)
    if g==0:
        return f**2
    elif g==1:
        return -f*(1-f)
    elif g==2:
        return (1-f)**2
    else:
        print(g)
        raise(ValueError)

def equal_effect_sim(nvar):
    return 2*rand.binomial(1,0.5,size=nvar)-1


def phenosim(add_gts,dom_gts,interaction_gts,h2,hd,h2_epi,equal_effects,var_effect):
    nc=add_gts.shape[1]
    N=add_gts.shape[0]
    # Generate additive effects
    if equal_effects:
        add_effects=equal_effect_sim(nc)
    else:
        add_effects=rand.randn((nc))
    # Generate Additive Component of Model
    A=np.dot(add_gts,add_effects)
    # Scale to appropriate variance
    A=np.sqrt(h2)*A/A.std()

    # Generate dominance effects
    if dom_gts.shape[0]>0:
        nd=dom_gts.shape[1]
        if equal_effects:
            dom_effects=equal_effect_sim(nd)
        else:
            dom_effects=rand.randn((nd))
        # Generate Dominance component of model
        D=np.dot(dom_gts,dom_effects)
        # Scale to have appropriate variance
        D=np.sqrt(hd)*D/D.std()
    else:
        D=0

    # Sample interaction effects
    if h2_epi>0:
        n_interact=interaction_gts.shape[1]
        if equal_effects:
            epi_effects=equal_effect_sim(n_interact)
        else:
            epi_effects=rand.randn((n_interact))
        Epi=np.dot(interaction_gts,epi_effects)
        Epi=np.sqrt(h2_epi)*Epi/Epi.std()
    else:
        Epi=0

    # Simulate residual error
    E=rand.randn((N))
    if var_effect > 0:
        var_effects = var_effect*equal_effect_sim(nc)
        V = np.exp(np.dot(add_gts, var_effects))
    else:
        V=np.ones((nc))
    E_var=(1-h2-h2_epi-hd)*V
    E=np.array([np.random.normal(scale=E_var[x]) for x in xrange(0,N)])

    # Combine to form phenotype
    Y=A+D+Epi+E
    return(Y)

# Find out number of chromosomes and genotypes in each chr
gfiles=open(genofiles,'r')
chr_lengths=[]
genofile=gfiles.readline().split('\n')[0]
nchr=0
while len(genofile)>0:
    ghdf5=h5py.File(genofile,'r')
    genotypes=np.transpose(ghdf5['genotypes'])
    N=genotypes.shape[0]
    chr_lengths.append(genotypes.shape[1])
    nchr+=1
    ghdf5.close()
    genofile=gfiles.readline().split('\n')[0]

gfiles.close()
genome_length=sum(chr_lengths)

if nc>genome_length:
	raise(ValueError("Number of loci, "+str(genome_length)+ ", less than number of causal loci in simulation, "+str(nc)))
print('Simulating Phenotype from '+str(genome_length)+' variants from '+
      str(nchr)+' chromosomes for '+str(N)+' individuals')
print('Minimum MAF: '+str(args.min_maf))
print('Max missingness: '+str(args.max_missing)+'%')

# Store causal loci by chromosome and index within chromosome gts
causal_loci=np.zeros((nc,2),dtype=int)
# Store causal gts in array
causal_gts=np.zeros((N,nc))
n_selected=0
gfiles=open(genofiles,'r')
for i in xrange(0,nchr):
    genofile=gfiles.readline().split('\n')[0]
    genofile=h5py.File(genofile,'r')
    genotypes=np.array(genofile['genotypes']).T
    # Get sample IDs
    if i==0:
        sample_id=np.array(genofile['sample_id'])
        sample_id_dict=id_dict_make(sample_id)
    else:
        sample_i=np.array(geno_file['sample_id'])
        sample_i_match = np.array([sample_id_dict[x] for x in sample_i])
        genotypes=genotypes[sample_i_match,:]
    # Number of causal variants from this chromosome
    if i==nchr-1:
	nc_chr=nc-n_selected
    else:
	nc_chr=int(math.floor(float(chr_lengths[i])/float(genome_length)*nc))
    if nc_chr>0:
    	# Skip this many columns between causal variants
        if chr_lengths[i]==nc_chr:
            colskip=1
        else:
            colskip=int(chr_lengths[i]/nc_chr)-1
    	# Record indices of causal variants within chromosome
    	causal_chr=np.array(range(0,chr_lengths[i],colskip),dtype=int)
    	causal_chr=causal_chr[0:nc_chr]
        # Select nearest locus with MAF greater than min_maf
        for l in xrange(0,nc_chr):
            loc_index=causal_chr[l]
            g=genotypes[:,loc_index]
            #print(loc_index)
            missing=100*float(np.sum(g<0))/float(genotypes.shape[0])
            freq=freq=np.mean(g[g>0])/2.0
            # Mean imputation
            g[g<0]=freq*2.0
            if freq>0.5:
                freq=1-freq
            # Check if it passes threshold
            while freq<args.min_maf or missing>args.max_missing:
                loc_index+=1
                g=genotypes[:,loc_index]
                missing=100*float(np.sum(g<0))/float(genotypes.shape[0])
                freq=np.mean(g[g>0])/2.0
                # Mean imputation
                g[g<0]=freq*2.0
                if freq>0.5:
                    freq=1-freq
            causal_chr[l]=loc_index
            print(loc_index)
    	# Store causal variant locations
    	causal_loci[n_selected:(n_selected+nc_chr),0]=i+1
    	causal_loci[n_selected:(n_selected+nc_chr),1]=causal_chr
    	# Add causal variants from this chromosome to the matrix of causal vars
        print('Grabbing genotypes')
    	causal_gts[:,n_selected:(n_selected+nc_chr)]=genotypes[:,causal_chr]
        print('Got genotypes')
	n_selected+=nc_chr
gfiles.close()
#code.interact(local=locals())
# Record causal loci in phenofile
phenofile=h5py.File(phenout,'w')
phenofile.create_dataset('causal',data=causal_loci)

####################
# Mean normalise variables
add_gts=causal_gts-causal_gts.mean(axis=0)

# Choose SNPs to have dominance effects
if nd>0:
    dom_indices=[i for i in xrange(0,nc)]
    dom_indices=np.array(rand.choice(dom_indices,size=nd,replace=False),dtype=int)
    dom_gts=np.array(causal_gts[:,dom_indices],dtype=float)
    freqs=np.mean(dom_gts,axis=1)/2.0
    for i in xrange(0,nd):
        freq=freqs[i]
        for j in xrange(0,N):
            dom_gts[j,i]=dom_convert(dom_gts[j,i],freq)

    # Convert indices relative to additive to indices in SNP file
    dom_indices=causal_loci[dom_indices,]
    phenofile.create_dataset('dom',data=dom_indices)
else:
    dom_gts=np.array(())

## Interactions
npairs=nc*(nc-1)/2
# Array for interaction genotypes
interaction_gts=np.zeros((N,n_interact))

# Sample pairs to have interactions
interact_indices=rand.choice(npairs,size=n_interact,replace=False)
# Record which SNPs interact
interacting_snp_indices=np.zeros((n_interact,4),dtype=int)

# Fill in interaction genotypes
pair_count=0
interact_count=0
for i in xrange(0,nc):
    for j in xrange(0,i):
        if pair_count in interact_indices:
            # Record which SNPs interact
            interacting_snp_indices[interact_count,0:2]=causal_loci[i,:]
            interacting_snp_indices[interact_count,2:4]=causal_loci[j,:]
	    # Form interaction genotypes
            interaction_gts[:,interact_count]=add_gts[:,i]*add_gts[:,j]
            interact_count+=1
        pair_count+=1

phenofile.create_dataset('interactions',data=interacting_snp_indices)
phenofile.create_dataset('sample_id',data=sample_id)

# Simulate phenotypes and save to sim group of screen hdf5 file
phenotypes=phenofile.create_dataset('phenotypes',(N,nphen),dtype=float)
print('Simulating Phenotypes')
if args.normal_effects:
    equal_effects=False
else:
    equal_effects=True
for i in xrange(0,nphen):
    if nd>0:
        phenotypes[:,i]=phenosim(add_gts,dom_gts,interaction_gts,h2,hd,h2_epi,equal_effects,args.variance_effect)
    else:
        phenotypes[:,i]=phenosim(add_gts,np.array(()),interaction_gts,h2,hd,h2_epi,equal_effects,args.variance_effect)

# Save causal genotypes (including dominance and epistasis)
if args.save_causal_gts:
	phenofile.create_dataset('causal_gts',data=causal_gts)
	phenofile.create_dataset('interaction_gts',data=interaction_gts)
	phenofile.create_dataset('dom_gts',data=dom_gts)
phenofile.close()