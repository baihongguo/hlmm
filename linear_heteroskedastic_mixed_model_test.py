__author__ = 'ay'

import numpy as np
import imp
from scipy.optimize import fmin_l_bfgs_b

import matplotlib.pyplot as plt
lhmm = imp.load_source('lhmm', '/Users/ay/Dropbox/Interaction_Detection/scripts/python/linear_heteroskedastic_mixed_model.py')


def gradient(pars,dx,y,X,V,G):
    nvars=len(pars)
    gradient=np.zeros((nvars))
    for i in xrange(0,nvars):
        delta=np.zeros(nvars)
        delta[i]=dx
        upper=pars+delta
        lower=pars-delta
        gradient[i]=(lhmm.likelihood_and_gradient(upper,y,X,V,G,True)-lhmm.likelihood_and_gradient(lower,y,X,V,G,True))/(2*dx)
    return gradient

def simulate_phenotype(n,nrep,h2,alpha,beta,X,V,G):
    rnd_effect=h2*G.dot(G.T)
    mean=X.dot(alpha)
    D=np.diag(np.exp(V.dot(beta)))
    # Simulate phenotype
    Sigma=rnd_effect+D
    return np.random.multivariate_normal(mean=mean,cov=Sigma,size=(nrep))

def fit_model(y,X,V,G):
    # Initial parameters
    n_fixed_variance=V.shape[1]
    init_pars=np.zeros((n_fixed_variance+1))
    init_pars[n_fixed_variance]=0.5
    # Parameter bounds
    parbounds=[]
    for i in xrange(0,n_fixed_variance):
        parbounds.append((None,None))
    parbounds.append((0.00001,None))
    # Fit model
    optimized=fmin_l_bfgs_b(func=lhmm.likelihood_and_gradient,x0=init_pars,args=(y,X,V,G,False),approx_grad=False,bounds=parbounds)
    # Get MLE
    alpha_opt=lhmm.alpha_mle_final(optimized[0],y, X, V, G)
    opt_pars=np.hstack((alpha_opt,optimized[0]))
    # Approximate Hessian
    par_cov=lhmm.parameter_covariance(opt_pars,y,X,V,G,1e-5)
    return [opt_pars,par_cov[0]]


######## Simulate phenotypes ########
n=2*10**3
h2=0

# Make random effect loci
l=10
G=np.random.normal(0,1,size=(n,l))
G=G/np.std(G,axis=0)
G=G-np.mean(G,axis=0)
G=np.power(l,-0.5)*G
rnd_effect=h2*G.dot(G.T)

# Make fixed effects for mean
X=np.ones((n,2))
X[:,1]=np.random.normal(0,1,size=n)
alpha=np.ones((2))
#alpha=np.zeros((2))

# Make fixed effects for variance
V=np.ones((n,2))
V[:,1]=np.random.normal(0,5,size=n)
beta=np.ones((2))/10
#beta=np.zeros((2))

# Simulate a phenotype
y=simulate_phenotype(n,1,h2,alpha,beta,X,V,G)[0,:]
plt.hist(y)

############ Test Likelihood Computation #################
## Get random parameters to test
pars=np.random.normal(0,1,size=3)
pars[2]=0.1
## Check likelihood computation
safe=lhmm.safe_likelihood(pars,y, X, V, G)
low_rank=lhmm.likelihood_and_gradient(pars,y, X,V,G,True)
np.allclose(safe,low_rank)

############# Check gradient computation #################
grad_num=gradient(pars,1e-5,y,X,V,G)
grad=lhmm.likelihood_and_gradient(pars,y,X,V,G,False)[1]
np.allclose(grad_num,grad)


############# Check Optimization ################
n_fixed_variance=2
init_pars=np.zeros((n_fixed_variance+1))
init_pars[2]=0.1
parbounds=[]
for i in xrange(0,n_fixed_variance):
    parbounds.append((None,None))
parbounds.append((0.00001,None))

lhmm = imp.load_source('lhmm', '/Users/ay/Dropbox/Interaction_Detection/scripts/python/linear_heteroskedastic_mixed_model.py')


null=fmin_l_bfgs_b(func=lhmm.likelihood_and_gradient,x0=init_pars,args=(y,X,V,G,False),approx_grad=False,bounds=parbounds)

lin_var=V.dot(null[0][0:2])
D=np.exp(lin_var)
y_scaled=(y-X.dot(np.zeros((2))))/np.sqrt(D)

null_safe=fmin_l_bfgs_b(func=lhmm.safe_likelihood,x0=init_pars,args=(y,X,V,G),approx_grad=True,bounds=parbounds)

np.allclose(null[0],null_safe[0])
np.allclose(null[1],null_safe[1])

############ Check Standard Error Computation ##########
# Simulate phenotypes
nrep=500
n_pars=len(alpha)+len(beta)+1
y=simulate_phenotype(n,nrep,h2,alpha,beta,X,V,G)

optimized_parameters=np.zeros((nrep,n_pars))
se_estimates=np.zeros((nrep,n_pars))
for i in xrange(0,nrep):
    fitted=fit_model(y[i,:],X,V,G)
    optimized_parameters[i,:]=fitted[0]
    se_estimates[i,:]=fitted[1]

mean_estimates=np.mean(optimized_parameters,axis=0)
std_resids=(optimized_parameters-1)/se_estimates
np.savetxt('std_resids.txt',std_resids)



## Check if standard errors from non-mixed model are good approx
lhm = imp.load_source('lhm', '/Users/ay/Dropbox/Interaction_Detection/scripts/python/linear_heteroskedastic_model.py')

alpha_se_ests=np.zeros((nrep,len(alpha)))
beta_se_ests=np.sqrt(np.diag(lhm.beta_cov(V))).reshape(1,len(beta))
beta_se_ests=beta_se_ests.repeat(nrep,axis=0)
for i in xrange(0,nrep):
    alpha_cov=lhm.alpha_cov(X,V,optimized_parameters[i,(len(alpha)):(len(alpha)+len(beta))])
    alpha_se_ests[i,0:len(alpha)]=np.sqrt(np.diag(alpha_cov))

np.savetxt('alpha_beta_ses.txt',se_estimates)
np.savetxt('alpha_approx_ses.txt',alpha_se_ests)
np.savetxt('beta_approx_ses.txt',beta_se_ests)

# Simulate regressing out other covariates
n=2*10**3
h2=1

# Make random effect loci
l=10
G=np.random.normal(0,1,size=(n,l))
G=G/np.std(G,axis=0)
G=G-np.mean(G,axis=0)
G=np.power(l,-0.5)*G
rnd_effect=h2*G.dot(G.T)

# Make fixed effects for mean
X=np.ones((n,3))
X[:,1]=np.random.normal(0,1,size=n)
X[:,2]=np.random.normal(0,1,size=n)
alpha=np.ones((3))
#alpha=np.zeros((2))

# Make fixed effects for variance
V=np.ones((n,3))
V[:,1]=np.random.normal(0,1,size=n)
V[:,2]=np.random.normal(0,1,size=n)
beta=np.ones((3))/10
#beta=np.zeros((2))

# Simulate a phenotype
y=simulate_phenotype(n,1,h2,alpha,beta,X,V,G)[0,:]
plt.hist(y)

n_fixed_variance=3
init_pars=np.zeros((n_fixed_variance+1))
init_pars[2]=0.1
parbounds=[]
for i in xrange(0,n_fixed_variance):
    parbounds.append((None,None))
parbounds.append((0.00001,None))

reduced=fmin_l_bfgs_b(func=lhmm.likelihood_and_gradient,x0=init_pars,
                      args=(y,X,V[:,0:2],G,False),approx_grad=False,bounds=parbounds)



# Simulate phenotypes
nrep=500
n_pars=len(alpha)+len(beta)+1
y=simulate_phenotype(n,nrep,h2,alpha,beta,X,V,G)

optimized_var_par=np.zeros((nrep))
var_par_se_estimates=np.zeros((nrep))
for i in xrange(0,nrep):
    # Fit reduced model
    fitted=fit_model(y[i,:],X,V[:,0:2],G)
    # Rescale
    alpha_mle=fitted[0][0:3]
    beta_mle=fitted[0][3:5]
    y_scaled=(y[i,:]-X.dot(alpha_mle))/np.exp(0.5*V[:,0:2].dot(beta_mle))
    # Fit scaled model
    scaled_fitted=fit_model(y_scaled,X,V[:,[0,2]],G)
    optimized_var_par[i]=fitted[0][4]
    se_estimates[i]=fitted[1][4]

std_resids=(optimized_var_par-0.1)/se_estimates
plt.hist(std_resids)
np.savetxt('std_resids_var_scaled.txt',std_resids)