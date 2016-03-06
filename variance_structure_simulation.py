import h5py
import numpy as np

def simulate_phenotype(alpha,beta,X,V):
    D=np.exp(V.dot(beta))
    Z=np.random.randn((X.shape[0]))
    Y=X.dot(alpha)+np.sqrt(D)*Z
    return Y

ukbb='/well/donnelly/ukbiobank_project_8874/'

chr22_f=h5py.File(ukbb+'flw/interim_genotype_data/calls/ukb8874.chr22.calls.transpose.caucasian.hdf5','r')

genotypes=np.transpose(np.array(chr22_f['genotypes'],dtype=int))

# Randomly sample rows
nc=100
N=genotypes.shape[0]
causal_loci=np.zeros((nc,2),dtype=int)
# Store causal gts in array
causal_gts=np.zeros((N,nc))
n_selected=0


# Skip this many columns between causal variants
colskip=int(genotypes.shape[1]/nc)-1
# Record indices of causal variants within chromosome
causal_indices=np.zeros((nc),dtype=int)
# Select nearest locus with MAF greater than min_maf
for l in xrange(0,nc):
    loc_index=colskip*l
    g=genotypes[:,loc_index]
    #print(loc_index)
    freq=freq=np.mean(g[g>0])/2.0
    missing=100*float(np.sum(g<0))/float(genotypes.shape[0])
    # Mean imputation
    g[g<0]=freq*2.0
    # Freq
    if freq>0.5:
        freq=1-freq
    while freq<0.10 or missing>1:
        loc_index+=1
        g=genotypes[:,loc_index]
        missing=100*float(np.sum(g<0))/float(genotypes.shape[0])
        freq=np.mean(g[g>0])/2.0
        # Mean imputation
        g[g<0]=freq*2.0
        if freq>0.5:
            freq=1-freq
    causal_indices[l]=loc_index
    causal_gts[:,l]=g
    #print(loc_index


# Get north east co-ordinates
north_east=np.genfromtxt(ukbb+'ay/brit_north_east.txt',dtype=float,
                         missing_values='NA')

# Generate effect sizes for genotypes
#alpha=np.random.randn((nc))
alpha=np.zeros((nc))

# Variance parameters
beta=np.array([0,0.2,-0.2])


# Generate effect sizes for north and east co-ordinates
north_east_mean_effects=np.array([1,-1])

y=simulate_phenotype(np.hstack((alpha,north_east_mean_effects)),
                     beta,np.hstack((causal_gts,north_east[:,1:3])),
                     np.hstack((np.ones((N,1)),north_east[:,1:3])))

y_f=h5py.File(ukbb+'ay/linear_variance/simulations/y_north_east.hdf5','w')
y_f['phenotypes']=y

y_f['north_east_var_effects']=beta[1:3]
y_f['north_east_mean_effects']=north_east_mean_effects

y_f.close()