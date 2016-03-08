import argparse,os, h5py, math
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('phenofile',type=str,help='Location of directory with phenotype hdf5')
parser.add_argument('outdir',type=str,help='name of phenotype: name.hdf5 in phenodir')
parser.add_argument('--block_size',type=int,help='size of block of SNPs to test at once',
                    default=2000)
parser.add_argument('--mean_covar',type=str,help='Location of .hdf5 file with matrix of fixed mean effect variables',
                    default=None)
parser.add_argument('--variance_covar',type=str,help='Locaiton of .hdf5 file with matrix of fixed variance effects',
                    default=None)
parser.add_argument('--block_list',type=str,help='Location of file with chr,block to do inference on',default='')
parser.add_argument('--fit_mean_covariates',action='store_true',help='Do not project out mean covariates',default=False)
parser.add_argument('--fit_variance_covariates',action='store_true',help='Do not project out variance covariates',default=False)
parser.add_argument('--dom',action='store_true',help='Fit dominance effect',default=False)
parser.add_argument('--gvar',action='store_true',help='Fit general variance effect',default=False)

args=parser.parse_args()


geno_prefix='/well/donnelly/ukbiobank_project_8874/ay/genotypes/chr.'
geno_suffix='.hdf5'

out_prefix=args.outdir

if len(args.block_list)>0:
    block_list=np.loadtxt(args.block_list,dtype=int)
    chr_block_list=dict()
    for i in xrange(0,block_list.shape[0]):
        chr_block_list[(int(block_list[i,0]),int(block_list[i,1]))]=i


#for chr in xrange(1,22):
for chr in xrange(21,23):
    # Chromosome string
    chr_string='%02d'%chr
    # Get chromosome length
    chr_hdf5=geno_prefix+str(chr)+geno_suffix
    print(chr_hdf5)
    chr_file=h5py.File(chr_hdf5,'r')
    chr_length=chr_file['genotypes'].shape[0]
    chr_file.close()
    # Divide into blocks
    n_blocks=chr_length/args.block_size
    block_lengths=[args.block_size]*n_blocks
    block_lengths.append(chr_length-args.block_size*n_blocks)
    n_blocks+=1
    print('Using '+str(n_blocks)+' blocks')
    n_block_digits=math.ceil(math.log(n_blocks,10))
    fmt='%0'+str(n_block_digits)+'d'
    # Send jobs for each block
    block_start=0
    for j in xrange(0,n_blocks):
    #for j in xrange(1,2):
        block_string=fmt%j
        block_end=block_start+block_lengths[j]
        out_suffix='chr.'+chr_string+'.'+block_string
        outfile=out_prefix+out_suffix
        err_file=out_prefix+'err/'+out_suffix
        out_file=out_prefix+'out/'+out_suffix
        command='qsub -e '+err_file+' -o '+out_file+' -v OMP_NUM_THREADS=1 -q short.qc '
        # Script
        command+='/well/donnelly/glmm/hlmm/linear_heteroskedastic_model.py '
        # Genofile
        command+=chr_hdf5+' '
        # Interval
        command+=str(block_start)+' '+str(block_end)+' '
        # Phenofile
        command+=args.phenofile+' '
        # Outprefix
        command+=outfile+' '
        if not args.mean_covar==None:
            command+='--mean_covar '+args.mean_covar+' '
        if not args.variance_covar==None:
            command+='--variance_covar '+args.variance_covar
        if args.fit_mean_covariates:
            command+=' --fit_mean_covariates'
        if args.fit_variance_covariates:
            command+=' --fit_variance_covariates'
        if args.dom:
            command+=' --dom'
        if args.dom:
            command+=' --gvar'
        # Check if in list
        if len(args.block_list)>0:
            chr_block=(chr,j)
            if chr_block in chr_block_list:
                os.system(command)
                print(command)
        else:
            os.system(command)
            print(command)
        block_start=block_end