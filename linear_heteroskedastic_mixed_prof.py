#!/apps/well/python/2.7.8/bin/python
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import chi2
from scipy import linalg
import h5py, argparse, code, imp
#import linear_heteroskedastic_model as lhm
lhm = imp.load_source('lhm', '/well/donnelly/glmm/hlmm/linear_heteroskedastic_model.py')

## Convert 0,1,2 genotypes to het indicator variables
def dom_convert(g):
    if g==1:
        return 1
    else:
        return 0

# Calculate -log10 pvals from chi square
def neglog10pval(x,df):
    return -np.log10(np.e)*chi2.logsf(x,df)

def safe_likelihood(pars,*args):
    y, X, V, G = args
    n_fixed_variance=V.shape[1]
    # Get parameters
    beta=pars[0:n_fixed_variance]
    h2=pars[n_fixed_variance]
    # Print
    beta_print=vector_print(beta)
    #print('h^2: '+str(round(h2,4))+'\t'+' beta: '+beta_print)
    ## Calculate common variables
    # heteroscedasticity
    Vb=np.dot(V,beta)
    D=np.diag(np.exp(Vb))
    G_cov=h2*G.dot(G.T)
    Sigma=D+G_cov
    # Calculate inverse and determinant
    Sigma_inv=linalg.inv(Sigma)
    Sigma_logdet=np.log(linalg.det(Sigma))
    ## Calculate alpha MLE
    A=np.dot(np.dot(np.transpose(X),Sigma_inv),X)
    b=np.dot(np.dot(np.transpose(X),Sigma_inv),y)
    if len(X.shape)==1:
        alpha=b/A
    else:
        alpha=linalg.solve(A,b)
    alpha_print=vector_print(alpha)
    #print('Alpha: '+alpha_print)
    ## Calculate likelihood
    resid=y-np.dot(X,alpha)
    L=Sigma_logdet+np.dot(np.dot(np.transpose(resid),Sigma_inv),resid)
    ## Print
    #print('Likelihood: '+str(round(-L,4))+'\t')
    return L

def full_grad_beta(pars,*args):
    y, X, V, G = args
    n_fixed_variance=V.shape[1]
    l=G.shape[1]
    n=G.shape[0]
    # Get parameters
    beta=pars[0:n_fixed_variance]
    h2=pars[n_fixed_variance]
    #
    Vb=np.dot(V,beta)
    D=np.diag(np.exp(Vb))
    G_cov=h2*G.dot(G.T)
    Sigma=D+G_cov
    # Calculate inverse and determinant
    Sigma_inv=linalg.inv(Sigma)
     ## Calculate alpha MLE
    A=np.dot(np.dot(np.transpose(X),Sigma_inv),X)
    b=np.dot(np.dot(np.transpose(X),Sigma_inv),y)
    if len(X.shape)==1:
        alpha=b/A
    else:
        alpha=linalg.solve(A,b)
    alpha_print=vector_print(alpha)
    #print('Alpha: '+alpha_print)
    ## Calculate likelihood
    resid=y-np.dot(X,alpha)
    ## Calculate empirical covariance
    resid=resid.reshape((n,1))
    S=resid.dot(resid.T)
    # Calculate matrix
    C=Sigma_inv-Sigma_inv.dot(S.dot(Sigma_inv))
    #print(C.shape)
    # Calculate gradient
    grad=np.zeros((n_fixed_variance))
    for i in xrange(0,n):
        grad+=C[i,i]*D[i,i]*V[i,:]
    return grad


def alpha_mle(h2,X_scaled,X,y,Z_scaled,Z,Lambda_inv):
    XtD=np.transpose(X_scaled)
    XtDZ=np.dot(XtD,Z)
    XtDZLambda=np.dot(XtDZ,Lambda_inv)
    X_cov_Z=np.dot(XtDZLambda,np.transpose(XtDZ))
    X_cov=np.dot(XtD,X)
    # Matrix multiplying alpha hat
    A=X_cov-h2*X_cov_Z
    # RHS
    X_cov_y=np.dot(XtD,y)
    Z_cov_y=np.dot(np.transpose(Z_scaled),y)
    X_cov_Z_cov_y=np.dot(XtDZLambda,Z_cov_y)
    b=X_cov_y-h2*X_cov_Z_cov_y
    if len(X.shape)==1:
        alpha=b/A
    else:
        alpha=linalg.solve(A,b)
    return alpha

def grad_h2_inner(Lambda_inv,Z_cov,Lambda_inv_rnd_resid):
    dl=0
    # Calculate trace
    dl+=np.sum(Lambda_inv*Z_cov)
    # Calculate inner product
    dl+=-np.sum(np.square(Lambda_inv_rnd_resid))
    return dl

@prof
def var_weight(h2,resid,G,Lambda_inv,Lambda_inv_rnd_resid):
    # Compute diagonal elements of nxn covariance matrix
    #cov_diagonal=np.einsum('ij,ij->i', np.dot(G, Lambda_inv), G)
    cov_diagonal=(G.dot(Lambda_inv) * G).sum(-1)
    # Compute weights coming from inner product in low rank space
    a=G.dot(Lambda_inv_rnd_resid)
    k=np.square(resid)+h2*cov_diagonal+h2*a*(h2*a-2*resid)
    return k

def linear_variance_approx_mle(V,h2,resid,G,Lambda_inv,Lambda_inv_rnd_resid):
    #print('Computing variance weights')
    k=var_weight(h2,resid,G,Lambda_inv,Lambda_inv_rnd_resid)
    #print('Solving linear system for approximate heteroskedasticity MLEs')
    V_T_scaled=np.transpose(V)*k
    A=np.dot(V_T_scaled,V)
    b=np.dot(np.transpose(V),k-1)
    if len(V.shape)==1:
        beta=b/A
    else:
        beta=linalg.solve(A,b)
    return beta

def init_beta(D_inv,h2,y,X,V,G):
    l=G.shape[1]
    # Low rank covariance
    G_scaled_T=np.transpose(G)*D_inv
    G_scaled=np.transpose(G_scaled_T)
    G_cov=np.dot(G_scaled_T,G)
    Lambda=np.identity(l,float)+h2*G_cov
    Lambda_inv=linalg.inv(Lambda)
    ## Calculate MLE of fixed effects
    X_scaled=np.transpose(np.transpose(X)*D_inv)
    alpha=alpha_mle(h2,X_scaled,X,y,G_scaled,G,Lambda_inv)
    ## Residuals
    resid=y-np.dot(X,alpha)
    rnd_resid=np.dot(np.transpose(G_scaled),resid)
    Lambda_inv_rnd_resid=np.dot(Lambda_inv,rnd_resid)
    # Get initial guess for fixed variance effects using linear approximation
    return linear_variance_approx_mle(V,h2,resid,G,Lambda_inv,Lambda_inv_rnd_resid)

def vector_print(alpha,digits=4):
    if len(alpha.shape)==0:
        alpha_print=str(round(alpha,digits))
    else:
        alpha_print=str([round(a,digits) for a in alpha ])
    return alpha_print

def vector_out(alpha,se,digits=4):
    ## Calculate test statistics
    # t-statistic
    t=alpha/se
    # chi-square statistic
    x2=np.square(t)
    # Create output strings
    if len(alpha.shape)==0:
        pval=neglog10pval(x2,1)
        alpha_print=str(round(alpha,digits))+'\t'+str(round(se,digits))+'\t'+str(round(t,digits))+'\t'+str(round(pval,digits))
    else:
        pvals=[neglog10pval(x,1) for x in x2]
        alpha_print=''
        for i in xrange(0,len(alpha)-1):
            alpha_print+=str(round(alpha[i],digits))+'\t'+str(round(se[i],digits))+'\t'+str(round(t[i],digits))+'\t'+str(round(pvals[i],digits))+'\t'
        i+=1
        alpha_print+=str(round(alpha[i],digits))+'\t'+str(round(se[i],digits))+'\t'+str(round(t[i],digits))+'\t'+str(round(pvals[i],digits))
    return alpha_print

def name_print(names):
    print_str=''
    n_names=len(names)
    for name_index in xrange(0,n_names-1):
        print_str+=names[name_index]+'\tse\tt\t-log10(p-value)\t'
    print_str+=names[n_names-1]+'\tse\tt\t-log10(p-value)\n'
    return print_str

# Only for positive semi-definite matrix
def inv_from_eig(eig):
    U_scaled=eig[1]*np.power(eig[0],-0.5)
    return np.dot(U_scaled,U_scaled.T)

@prof
def likelihood_and_gradient(pars,*args):
    y, X, V, G, approx_grad = args
    l=G.shape[1]
    n=G.shape[0]
    if V.ndim==2:
        n_fixed_variance=V.shape[1]
    elif V.ndim==1:
        n_fixed_variance=1
        V=V.reshape((n,1))
    else:
        raise(ValueError('Incorrect dimension of Variance Covariate Array'))
    if X.ndim==2:
        n_fixed_mean=X.shape[1]
    elif X.ndim==1:
        n_fixed_mean=1
        X=X.reshape((n,1))
    else:
        raise(ValueError('Incorrect dimension of Mean Covariate Array'))
    # Get parameters
    beta=pars[0:n_fixed_variance]
    h2=pars[n_fixed_variance]
    # Print
    beta_print=vector_print(beta)
    #print('h^2: '+str(round(h2,4))+'\t'+' beta: '+beta_print)
    ## Calculate common variables
    # heteroscedasticity
    Vb=np.dot(V,beta)
    D_inv=np.exp(-Vb)
    # Low rank covariance
    G_scaled_T=np.transpose(G)*D_inv
    G_scaled=np.transpose(G_scaled_T)
    G_cov=np.dot(G_scaled_T,G)
    Lambda=np.identity(l,float)+h2*G_cov
    Lambda=linalg.eigh(Lambda,overwrite_a=True,turbo=True)
    logdet_Lambda=np.sum(np.log(Lambda[0]))
    Lambda_inv=inv_from_eig(Lambda)
    ## Calculate MLE of fixed effects
    X_scaled=np.transpose(np.transpose(X)*D_inv)
    alpha=alpha_mle(h2,X_scaled,X,y,G_scaled,G,Lambda_inv)
    alpha_print=vector_print(alpha)
    #print('Alpha: '+alpha_print)
    ## Residuals
    if n_fixed_mean>1:
        resid=y-np.dot(X,alpha)
    else:
        yhat=X*alpha
        yhat=yhat.reshape(y.shape)
        resid=y-yhat
    resid_square=np.square(resid)
    rnd_resid=np.dot(G_scaled_T,resid)
    Lambda_inv_rnd_resid=np.dot(Lambda_inv,rnd_resid)
    ### Calculate likelihood
    L=np.sum(Vb)+np.sum(resid_square*D_inv)+logdet_Lambda-h2*np.dot(np.transpose(rnd_resid),Lambda_inv_rnd_resid)
    #print('Likelihood: '+str(round(-L,4))+'\n')
    ### Calculate gradient
    if not approx_grad:
        grad=np.zeros((len(pars)))
        # Calculate gradient with respect to beta
        k=var_weight(h2,resid,G,Lambda_inv,Lambda_inv_rnd_resid)
        n1t=np.ones((n)).reshape((1,n))
        grad[0:n_fixed_variance]=np.dot(n1t,np.transpose(np.transpose(V)*(1-k*D_inv)))
        # Calculate gradient with respect to h2
        grad[n_fixed_variance]=grad_h2_inner(Lambda_inv,G_cov,Lambda_inv_rnd_resid)
        return L,grad
    else:
        return L

def grad_alpha(resid,X_grad_alpha):
    return -2*np.dot(resid.T,X_grad_alpha)

@profile
def grad_beta(h2,G_scaled_T,V,D_inv,resid,Lambda_inv):
    n=V.shape[0]
    # Low rank covariance
    rnd_resid=np.dot(G_scaled_T,resid)
    Lambda_inv_rnd_resid=np.dot(Lambda_inv,rnd_resid)
    ### Calculate likelihood
    # Get k variance weights
    k=var_weight(h2,resid,G,Lambda_inv,Lambda_inv_rnd_resid)
    n1t=np.ones((n)).reshape((1,n))
    return np.dot(n1t,np.transpose(np.transpose(V)*(1-k*D_inv)))

def grad_h2(G_scaled_T,G_cov,resid,Lambda_inv):
    rnd_resid=G_scaled_T.dot(resid)
    Lambda_inv_rnd_resid=Lambda_inv.dot(rnd_resid)
    return grad_h2_inner(Lambda_inv,G_cov,Lambda_inv_rnd_resid)

def Lambda_calc(G_scaled_T,h2,V):
    Lambda=np.identity(G_scaled_T.shape[0],float)+h2*G_scaled_T.dot(G)
    return Lambda

@prof
def parameter_covariance(pars,y,X,V,G,dx):
    # Pars: alpha, beta, h2
    n_fixed_mean=X.shape[1]
    n_fixed_variance=V.shape[1]
    n_pars=len(pars)
    l=G.shape[1]
    # Calculate intermediate variables
    alpha=pars[0:n_fixed_mean]
    resid=(y-X.dot(alpha))
    # Residual Error
    beta=pars[n_fixed_mean:(n_fixed_variance+n_fixed_mean)]
    D_inv=np.exp(-V.dot(beta))
    G_scaled_T=(G.T)*D_inv
    # Random Effect
    h2=pars[n_fixed_variance+n_fixed_mean]
    Lambda=Lambda_calc(G_scaled_T,h2,V)
    Lambda_inv=linalg.inv(Lambda,overwrite_a=True,check_finite=False)
    G_cov=G_scaled_T.dot(G)
    # Components of alpha gradient calculation
    X_scaled=np.transpose((X.T)*D_inv)
    X_grad_alpha=X_scaled-h2*np.dot(np.dot(G_scaled_T.T,Lambda_inv),G_scaled_T.dot(X))
    H=np.zeros((n_pars,n_pars))
    # Calculate alpha components of hessian
    for p in xrange(0,n_fixed_mean):
        # Calculate change in alpha gradient
        d=np.zeros((n_fixed_mean))
        d[p]=dx
        resid_upper=(y-X.dot(alpha+d))
        resid_lower=(y-X.dot(alpha-d))
        H[0:n_fixed_mean,p]=(grad_alpha(resid_upper,X_grad_alpha)-grad_alpha(resid_lower,X_grad_alpha))/(2.0*dx)
        # Calculate change in beta gradient
        H[n_fixed_mean:(n_pars-1),p]=(grad_beta(h2,G_scaled_T,V,D_inv,resid_upper,Lambda_inv)-grad_beta(h2,G_scaled_T,V,D_inv,resid_lower,Lambda_inv))/(2.0*dx)
        H[p,n_fixed_mean:(n_pars-1)]=H[n_fixed_mean:(n_fixed_mean+n_fixed_variance),p]
        # Calculate change in h2 gradient
        H[n_pars-1,p]=(grad_h2(G_scaled_T,G_cov,resid_upper,Lambda_inv)-grad_h2(G_scaled_T,G_cov,resid_lower,Lambda_inv))/(2.0*dx)
        H[p,n_pars-1]=H[n_pars-1,p]
    # Calculate beta components of Hessian
    for p in xrange(n_fixed_mean,n_pars-1):
        d=np.zeros((n_fixed_variance))
        d[p-n_fixed_mean]=dx
        # Changed matrices
        D_inv_upper=np.exp(-V.dot(beta+d))
        D_inv_lower=np.exp(-V.dot(beta-d))
        G_scaled_T_upper=(G.T)*D_inv_upper
        G_scaled_T_lower=(G.T)*D_inv_lower
        G_cov_upper=np.dot(G_scaled_T_upper,G)
        G_cov_lower=np.dot(G_scaled_T_lower,G)
        Lambda_inv_upper=linalg.inv(np.identity(l,float)+h2*G_cov_upper,overwrite_a=True,check_finite=False)
        Lambda_inv_lower=linalg.inv(np.identity(l,float)+h2*G_cov_lower,overwrite_a=True,check_finite=False)
        # Change in beta gradient
        H[n_fixed_mean:(n_pars-1),p]=(grad_beta(h2,G_scaled_T_upper,V,D_inv_upper,resid,Lambda_inv_upper)-grad_beta(h2,G_scaled_T_lower,V,D_inv_lower,resid,Lambda_inv_lower))/(2.0*dx)
        # Change in h2 gradient
        H[n_pars-1,p]=(grad_h2(G_scaled_T_upper,G_cov_upper,resid,Lambda_inv_upper)-grad_h2(G_scaled_T_lower,G_cov_lower,resid,Lambda_inv_lower))/(2.0*dx)
        H[p,n_pars-1]=H[n_pars-1,p]
    # Calculate h2 components of the Hessian
    Lambda_inv_upper=linalg.inv(np.identity(l,float)+(h2+dx)*G_cov,overwrite_a=True,check_finite=False)
    Lambda_inv_lower=linalg.inv(np.identity(l,float)+(h2-dx)*G_cov,overwrite_a=True,check_finite=False)
    H[n_pars-1,n_pars-1]=(grad_h2(G_scaled_T,G_cov,resid,Lambda_inv_upper)-grad_h2(G_scaled_T,G_cov,resid,Lambda_inv_lower))/(2.0*dx)
    par_cov=linalg.inv(0.5*H,overwrite_a=True,check_finite=False)
    par_se=np.sqrt(np.diag(par_cov))
    #code.interact(local=locals())
    return [par_se,par_cov]

def parameter_covariance_old(pars,y,X,V,G,dx):
    # Pars: alpha, beta, h2
    n_fixed_mean=X.shape[1]
    n_fixed_variance=V.shape[1]
    n_pars=len(pars)
    l=G.shape[1]
    # Calculate intermediate variables
    alpha=pars[0:n_fixed_mean]
    resid=(y-X.dot(alpha))
    # Residual Error
    beta=pars[n_fixed_mean:(n_fixed_variance+n_fixed_mean)]
    D_inv=np.exp(-V.dot(beta))
    G_scaled_T=(G.T)*D_inv
    # Random Effect
    h2=pars[n_fixed_variance+n_fixed_mean]
    Lambda=Lambda_calc(beta,h2,V,G)
    Lambda_inv=linalg.inv(Lambda)
    G_cov=G_scaled_T.dot(G)
    # Components of alpha gradient calculation
    X_scaled=np.transpose((X.T)*D_inv)
    X_grad_alpha=X_scaled-h2*np.dot(np.dot(G_scaled_T.T,Lambda_inv),G_scaled_T.dot(X))
    H=np.zeros((n_pars,n_pars))
    # Calculate alpha components of hessian
    for p in xrange(0,n_fixed_mean):
        # Calculate change in alpha gradient
        d=np.zeros((n_fixed_mean))
        d[p]=dx
        resid_upper=(y-X.dot(alpha+d))
        resid_lower=(y-X.dot(alpha-d))
        H[0:n_fixed_mean,p]=(grad_alpha(resid_upper,X_grad_alpha)-grad_alpha(resid_lower,X_grad_alpha))/(2*dx)
        # Calculate change in beta gradient
        H[n_fixed_mean:(n_pars-1),p]=(grad_beta(h2,G,V,D_inv,resid_upper,Lambda_inv)-grad_beta(h2,G,V,D_inv,resid_lower,Lambda_inv))/(2*dx)
        H[p,n_fixed_mean:(n_pars-1)]=H[n_fixed_mean:(n_fixed_mean+n_fixed_variance),p]
        # Calculate change in h2 gradient
        H[n_pars-1,p]=(grad_h2(h2,G,V,D_inv,G_cov,resid_upper,Lambda_inv)-grad_h2(h2,G,V,D_inv,G_cov,resid_lower,Lambda_inv))/(2*dx)
        H[p,n_pars-1]=H[n_pars-1,p]
    # Calculate beta components of Hessian
    for p in xrange(n_fixed_mean,n_pars-1):
        d=np.zeros((n_fixed_variance))
        d[p-n_fixed_mean]=dx
        # Upper matrices
        D_inv_upper=np.exp(-V.dot(beta+d))
        G_cov_upper=np.dot((G.T)*D_inv_upper,G)
        Lambda_inv_upper=linalg.inv(np.identity(l,float)+h2*G_cov_upper)
        # Lower matrices
        D_inv_lower=np.exp(-V.dot(beta-d))
        G_cov_lower=np.dot((G.T)*D_inv_lower,G)
        Lambda_inv_lower=linalg.inv(np.identity(l,float)+h2*G_cov_lower)
        # Change in beta gradient
        H[n_fixed_mean:(n_pars-1),p]=(grad_beta(h2,G,V,D_inv_upper,resid,Lambda_inv_upper)-grad_beta(h2,G,V,D_inv_lower,resid,Lambda_inv_lower))/(2*dx)
        # Change in h2 gradient
        H[n_pars-1,p]=(grad_h2(h2,G,V,D_inv_upper,G_cov_upper,resid,Lambda_inv_upper)-grad_h2(h2,G,V,D_inv_lower,G_cov_lower,resid,Lambda_inv_lower))/(2*dx)
        H[p,n_pars-1]=H[n_pars-1,p]
    # Calculate h2 components of the Hessian
    Lambda_inv_upper=linalg.inv(np.identity(l,float)+(h2+dx)*G_cov_upper)
    Lambda_inv_lower=linalg.inv(np.identity(l,float)+(h2-dx)*G_cov_upper)
    H[n_pars-1,n_pars-1]=(grad_h2(h2+dx,G,V,D_inv,G_cov,resid,Lambda_inv_upper)-grad_h2(h2-dx,G,V,D_inv,G_cov,resid,Lambda_inv_lower))/(2*dx)
    par_cov=linalg.inv(0.5*H)
    par_se=np.sqrt(np.diag(par_cov))
    return [par_se,par_cov]


def approx_hessian(pars,y,X,V,G,dx):
    # Pars: alpha, beta, h2
    n_fixed_mean=X.shape[1]
    n_fixed_variance=V.shape[1]
    n_pars=len(pars)
    H=np.zeros((n_pars,n_pars))
    for p in xrange(0,n_pars):
        d=np.zeros((n_pars))
        d[p]=dx
        pars_lower=pars-d
        pars_upper=pars+d
        # Alpha
        H[p,0:n_fixed_mean]=(grad_alpha(pars_upper,y,X,V,G)-grad_alpha(pars_lower,y,X,V,G))/(2*dx)
        # Beta
        H[p,n_fixed_mean:(n_fixed_mean+n_fixed_variance)]=(grad_beta(pars_upper,y,X,V,G)-grad_beta(pars_lower,y,X,V,G))/(2*dx)
        # h2
        H[p,n_pars-1]=(grad_h2(pars_upper,y,X,V,G)-grad_h2(pars_lower,y,X,V,G))/(2*dx)
    return H

def approx_par_cov(pars,y,X,V,G,dx):
    H=0.5*approx_hessian(pars,y,X,V,G,dx)
    par_cov=linalg.inv(H)
    par_se=np.sqrt(np.diag(par_cov))
    return [par_se,par_cov]

@profile
def alpha_mle_final(pars,*args):
    y, X, V, G = args
    n_fixed_variance=V.shape[1]
    l=G.shape[1]
    # Get parameters
    beta=pars[0:n_fixed_variance]
    h2=pars[n_fixed_variance]
    ## Calculate common variables
    # heteroscedasticity
    Vb=np.dot(V,beta)
    D_inv=np.exp(-Vb)
    # Low rank covariance
    G_scaled=np.transpose(G.T*D_inv)
    G_cov=np.dot(np.transpose(G),G_scaled)
    Lambda=np.identity(l,float)+h2*G_cov
    Lambda_inv=linalg.inv(Lambda)
    ## Calculate MLE of fixed effects
    X_scaled=np.transpose(X.T*D_inv)
    return alpha_mle(h2,X_scaled,X,y,G_scaled,G,Lambda_inv)

@profile
def learn_models_chr(args):
    # Get genotype matrix
    genotypes=args.genotypes
    if genotypes.ndim==1:
        chr_length=1
    else:
        chr_length=genotypes.shape[1]
    n=genotypes.shape[0]
    # Read in random effect
    if isinstance(args.selected_genotypes, basestring):
        Gfile=h5py.File(args.selected_genotypes,'r')
        G=np.array(Gfile['selected_genotypes'])
        Gfile.close()
    else:
        G = args.selected_genotypes
    if np.min(G)>0:
        raise(ValueError('Missing values in random effect genotypes'))
    l=float(G.shape[1])
    print(str(int(l))+' loci in random effect')
    # Get phenotype
    y=args.phenotype
    if len(y.shape)==2:
        y=y[:,0]
    n=float(n)
    y_not_nan=np.logical_not(np.isnan(y))
    # Remove phenotype NAs
    if np.sum(y_not_nan)<n:
        y=y[y_not_nan]
        # Remove NAs from genotypes
        genotypes=genotypes[y_not_nan,:]
        n=genotypes.shape[0]
        # and random effects
        G=G[y_not_nan,:]
        ## Get genotype matrix for random effect
    #code.interact(local=locals())
    print(str(n)+' non missing cases from phenotype')
    if 'fixed_mean' in args:
        # Normalise
        args.fixed_mean=args.fixed_mean[y_not_nan,:]
        fixed_mean_stds=np.std(args.fixed_mean,axis=0)
        args.fixed_mean=args.fixed_mean/fixed_mean_stds
                # Remove NAs from fixed effects
        n_fixed_mean=args.fixed_mean.shape[1]+1
        fixed_mean=np.zeros((n,n_fixed_mean))
        fixed_mean[:,0]=np.ones(n)
        fixed_mean[:,1:n_fixed_mean]=args.fixed_mean
        fixed_mean_names=np.zeros((n_fixed_mean),dtype='S10')
        fixed_mean_names[0]='Intercept'
        fixed_mean_names[1:n_fixed_mean]=args.fixed_mean_names
    else:
        fixed_mean=np.ones((n,1))
        n_fixed_mean=1
        fixed_mean_names=np.array(['Intercept'])
    if 'fixed_variance' in args:
        # Normalise
        args.fixed_variance=args.fixed_variance[y_not_nan,:]
        fixed_var_stds=np.std(args.fixed_variance,axis=0)
        args.fixed_variance=args.fixed_variance/fixed_var_stds
        # Add scale
        n_fixed_variance=args.fixed_variance.shape[1]+1
        fixed_variance=np.zeros((n,n_fixed_variance))
        fixed_variance[:,0]=np.ones(n)
        fixed_variance[:,1:n_fixed_variance]=args.fixed_variance
        fixed_variance_names=np.zeros((n_fixed_variance),dtype='S10')
        fixed_variance_names[0]='Intercept'
        fixed_variance_names[1:n_fixed_variance]=args.fixed_variance_names
    else:
        fixed_variance=np.ones((n,1))
        n_fixed_variance=1
        fixed_variance_names=np.array(['Scale'])
    n_pars=n_fixed_mean+n_fixed_variance+1
    print(str(n_pars)+' parameters in model')
    # Rescale
    G=np.power(l,-0.5)*G
    # Get initial value for h2
    h2_init=args.h2_init
    ######### Initialise output files #######
    ## Output file
    outfile=open(args.outprefix+'.models.gz','wb')
    outfile.write('SNP_index\tfrequency\tmean_llr\tmean_effect\tmean_effect_se\tmean_effect_t\tmean_effect_pval\t'
                  'var_llr\t''mean_effect_av\tmean_effect_av_se\tmean_effect_av_t\tmean_effect_av_pval\t'
                  'var_effect\tvar_effect_se\tvar_effect_t\tvar_effect_pval\n')
    ######### Fit Null Model ##########
    ## Get initial guesses for null model
    print('Fitting Null Model')
    # Assume no heteroskedasticity to start with
    # Get initial guess for fixed variance effects using linear approximation
    #beta_null=init_beta(D_inv_init,h2_init,y,fixed_mean,fixed_variance,G)
    # Optimize null model
    init_params=np.zeros((n_fixed_variance+1))
    init_model=lhm.optimize_model(y,fixed_mean,fixed_variance)
    init_params[0:n_fixed_variance]=init_model['beta']
    init_params[n_fixed_variance]=h2_init
    parbounds=[]
    for i in xrange(0,n_fixed_variance):
        parbounds.append((None,None))
    parbounds.append((0.00001,None))
    parbounds_av=[(None,None)]+parbounds
    null=fmin_l_bfgs_b(func=likelihood_and_gradient,x0=init_params,
                                args=(y, fixed_mean, fixed_variance, G, args.approx_grad),
                                approx_grad=args.approx_grad,
                                bounds=parbounds)
    ## Record fitting of null model
    # log-likelihood of null model
    null_ll=-0.5*(null[1]+n*np.log(2*np.pi))
    # mean effects
    alpha_null=alpha_mle_final(null[0],y, fixed_mean, fixed_variance, G)
    beta_null=null[0][0:n_fixed_variance]
    null_h2=null[0][n_fixed_variance]
    null_mle=np.hstack((alpha_null,null[0]))
    print('Calculating Standard Errors')
    if args.full_cov:
         null_mle_se=parameter_covariance(null_mle,y,fixed_mean,fixed_variance,G,1e-6)[0]
    else:
         null_mle_se=np.zeros((n_fixed_mean+n_fixed_variance))
         null_mle_se[0:n_fixed_mean]=np.sqrt(np.diag(lhm.alpha_cov(fixed_mean,fixed_variance,beta_null)))
         null_mle_se[n_fixed_mean:(n_fixed_mean+n_fixed_variance)]=np.sqrt(np.diag(lhm.beta_cov(fixed_variance)))
    # Get print out for fixed mean effects
    alpha_out=np.zeros((n_fixed_mean,2))
    alpha_out[:,0]=alpha_null
    alpha_out[:,1]=null_mle_se[0:n_fixed_mean]
    np.savetxt(args.outprefix+'.null_mean_effects.txt',
                              np.hstack((fixed_mean_names.reshape((n_fixed_mean,1)),alpha_out)),
                              delimiter='\t',fmt='%s')
    # variance effects
    beta_out=np.zeros((n_fixed_variance,2))
    beta_out[0:n_fixed_variance,0]=beta_null
    beta_out[0:n_fixed_variance,1]=null_mle_se[n_fixed_mean:(n_fixed_mean+n_fixed_variance)]
    np.savetxt(args.outprefix+'.null_variance_effects.txt',
                              np.hstack((fixed_variance_names.reshape((n_fixed_variance,1)),beta_out)),
                              delimiter='\t',fmt='%s')
    # variance parameter
    if args.full_cov:
        np.savetxt(args.outprefix+'.null_h2.txt',
                   np.array([null_ll,null_h2,null_mle_se[n_fixed_variance]]),
                   delimiter='\t',fmt='%s')
    else:
        np.savetxt(args.outprefix+'.null_h2.txt',
                   np.array([null_ll,null_h2,np.nan]),
                   delimiter='\t',fmt='%s')
    # variance of random effect
    # If not fitting covariates for each locus, reformulate as residuals of null model
    if not args.fit_mean_covariates:
        # Residual phenotype
        y=y-fixed_mean.dot(alpha_null)
        # Reformulate fixed_effects
        fixed_mean=np.ones((n,1))
        n_fixed_mean=1
    if not args.fit_variance_covariates:
        #code.interact(local=locals())
        # Residual phenotype
        lin_var_null=fixed_variance.dot(null[0][0:n_fixed_variance])
        D_null_sqrt=np.exp(0.5*lin_var_null)
        y=y/D_null_sqrt
        # Reformulate fixed_effects
        fixed_variance=np.ones((n,1))
        n_fixed_variance=1
        # Remake parbounds
        parbounds=[]
        for i in xrange(0,n_fixed_variance):
            parbounds.append((None,None))
        parbounds.append((0.00001,None))
        parbounds_av=[(None,None)]+parbounds
    ## Loop through loci
    for loc in xrange(0,chr_length):
        print(loc)
        likelihoods=np.array(['NaN','NaN','NaN'],dtype=float)
        llrs=np.array(['NaN','NaN'],dtype=float)
        allele_frq=np.nan
        # Filler for output if locus doesn't pass threshold
        additive_out='NaN\tNaN\tNaN\tNaN'
        additive_av_out='NaN\tNaN\tNaN\tNaN'
        variance_out='NaN\tNaN\tNaN\tNaN'
        test_gts=genotypes[:,loc]
        # Find missingness and allele freq
        test_gt_not_na=test_gts>=0
        n_missing=float(np.sum(np.logical_not(test_gt_not_na)))
        if n_missing<n:
            missingness=100.0*(n_missing/n)
            test_gts=test_gts[test_gt_not_na]
            allele_frq=np.mean(test_gts)/2
            if allele_frq>0.5:
                allele_frq=1-allele_frq
            if allele_frq>args.min_maf and missingness<args.max_missing:
                # Remove missing data rows
                y_l=y[test_gt_not_na]
                n_l=len(y_l)
                X_l=fixed_mean[test_gt_not_na,:]
                V_l=fixed_variance[test_gt_not_na,:]
                G_l=G[test_gt_not_na,:]
                ## Fit Null model ##
                print('Fitting locus null model')
                init_params=np.zeros((n_fixed_variance+1))
                init_params[0:n_fixed_variance]=lhm.optimize_model(y_l,X_l,V_l)['beta']
                init_params[n_fixed_variance]=null_h2
                null_l=fmin_l_bfgs_b(func=likelihood_and_gradient,x0=init_params,
                                args=(y_l, X_l, V_l, G_l, args.approx_grad),
                                bounds=parbounds)
                likelihoods[0]=-0.5*(null_l[1]+n_l*np.log(2*np.pi))
                h2_null=null_l[0][n_fixed_variance]
                ## Fit linear mean model ##
                print('Fitting locus linear model')
                test_gts=np.array(test_gts).reshape((n_l,1))
                X_l=np.hstack((X_l,test_gts))
                # Calculate initial parameters
                init_params[0:n_fixed_variance]=lhm.optimize_model(y_l,X_l,V_l)['beta']
                init_params[n_fixed_variance]=h2_null
                additive=fmin_l_bfgs_b(func=likelihood_and_gradient,x0=init_params,
                                args=(y_l, X_l, V_l, G_l, args.approx_grad),
                                bounds=parbounds)
                likelihoods[1]=-0.5*(additive[1]+n_l*np.log(2*np.pi))
                alpha_additive=alpha_mle_final(additive[0],y_l, X_l, V_l, G_l)
                beta_additive=additive[0][0:n_fixed_variance]
                # Estimate standard errors
                if args.full_cov:
                     additive_pars=np.hstack((alpha_additive,additive[0]))
                     additive_par_cov=parameter_covariance(additive_pars,y_l, X_l, V_l, G_l,1e-6)
                     additive_se=additive_par_cov[0]
                else:
                     additive_se=np.sqrt(np.diag(lhm.alpha_cov(X_l,V_l,beta_additive)))
                additive_out=vector_out(alpha_additive[n_fixed_mean],additive_se[n_fixed_mean],6)
                # Variance parameters
                h2_add=additive[0][n_fixed_variance]
                ## Fit linear mean and variance model ##
                print('Fitting locus linear mean and log-variance model')
                V_l=np.hstack((V_l,test_gts))
                init_params=np.zeros((n_fixed_variance+2))
                init_params[0:(n_fixed_variance+1)]=lhm.optimize_model(y_l,X_l,V_l)['beta']
                init_params[n_fixed_variance+1]=h2_add
                # Add to parbounds
                av=fmin_l_bfgs_b(func=likelihood_and_gradient,x0=init_params,
                                args=(y_l, X_l, V_l, G_l, args.approx_grad),
                                bounds=parbounds_av)
                # Likelihood
                likelihoods[2]=-0.5*(av[1]+n_l*np.log(2*np.pi))
                # Mean effect of locus
                alpha_av=alpha_mle_final(av[0],y_l, X_l, V_l, G_l)
                # Approximate standard errors
                av_se=np.zeros((2))
                if args.full_cov:
                    av_mle=np.hstack((alpha_av,av[0]))
                    av_par_cov=parameter_covariance(av_mle,y_l, X_l, V_l, G_l,1e-6)
                    av_se[0]=av_par_cov[0][n_fixed_mean]
                    av_se[1]=av_par_cov[0][n_fixed_mean+n_fixed_variance+1]
                else:
                    alpha_av_se=np.sqrt(np.diag(lhm.alpha_cov(X_l,V_l,av[0][0:(n_fixed_variance+1)])))
                    av_se[0]=alpha_av_se[n_fixed_mean]
                    beta_av_se=np.sqrt(np.diag(lhm.beta_cov(V_l)))
                    av_se[1]=beta_av_se[n_fixed_variance]
                additive_av_out=vector_out(alpha_av[n_fixed_mean],av_se[0],6)
                # Variance effect of locus
                variance_out=vector_out(av[0][n_fixed_variance],av_se[1])
                ## Write output ##
                # Chi-square statistics
                llrs[0]=2*(likelihoods[1]-likelihoods[0])
                llrs[1]=2*(likelihoods[2]-likelihoods[1])
            # General association
        outfile.write(str(args.start+loc) + '\t' + str(allele_frq)+'\t'+str(llrs[0])+'\t'+additive_out+
                      '\t'+str(llrs[1])+'\t'+additive_av_out+'\t'+variance_out+'\n')
    outfile.close()
    return

def id_dict_make(ids):
    id_dict={}
    for id_index in xrange(0,len(ids)):
        id_dict[ids[id_index]]=id_index
    return id_dict

if __name__ == "__main__":
    ######## Parse Arguments #########
    parser=argparse.ArgumentParser()
    parser.add_argument('genofile',type=str,help='Location of the .hdf5 file with genotypes as dataset')
    parser.add_argument('start',type=int,help='Index of locus in genofile from which to start computing test stats')
    parser.add_argument('end',type=int,help='Index of locus in genofile at which to finish computing test stats')
    parser.add_argument('phenofile',type=str,help='Location of the .hdf5 file with phenotypes as dataset')
    parser.add_argument('random_gts',type=str,help='Location of the .hdf5 file with the genotypes of the random effect')
    parser.add_argument('outprefix',type=str,help='Location to output csv file with test statistics')
    parser.add_argument('--mean_covar',type=str,help='Location of .hdf5 file with matrix of fixed mean effect variables',
                        default=None)
    parser.add_argument('--variance_covar',type=str,help='Locaiton of .hdf5 file with matrix of fixed variance effects',
                        default=None)
    parser.add_argument('--h2_init',type=float,help='Initial value for variance explained by random effect (default 0.05)',
                        default=0.05)
    parser.add_argument('--phen_index',type=int,help='If phenotype file contains multiple phenotypes, which row to choose (default 0)',
                        default=0)
    parser.add_argument('--approx_grad',action='store_false',default=False)
    parser.add_argument('--fit_mean_covariates',action='store_true',default=False)
    parser.add_argument('--fit_variance_covariates',action='store_true',default=False)
    parser.add_argument('--min_maf',type=float,help='Minimum minor allele frequency',default=0.05)
    parser.add_argument('--max_missing',type=float,help='Maximum percent of missing genotype calls',default=5)
    parser.add_argument('--full_cov',action='store_true',default=False)

    args=parser.parse_args()

    # Get test loci
    test_chr=h5py.File(args.genofile,'r')
    test_gts=test_chr['genotypes']
    # select subset to test
    if args.end>test_gts.shape[0]:
        args.end=test_gts.shape[0]
    args.genotypes=np.transpose(np.array(test_gts[args.start:args.end,:]))
    print('Number of test loci: '+str(args.genotypes.shape[1]))
    # Get sample ids
    geno_ids=np.array(test_chr['sample_id'])


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
    random_gts_f=h5py.File(args.random_gts,'r')
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