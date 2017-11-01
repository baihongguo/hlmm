import numpy as np
from scipy import linalg
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import zscore

from hlmm import hetlm

"""
Module for defining heteroskedastic linear models, simulating heteroskedastic linear models,
and finding maximum likelihood estimates of the parameters. 

Class: model
"""

class model(object):
    """
    Define a heteroskedastic linear mixed model and calculate likelihood, gradients, and maximum likelihood estimates of
    parameters.
    :param y: (n) or (n,1) numpy array of phenotype observations
    :param X: (n) or (n,c) numpy array of n observations of variables. This is the design matrix for the fixed mean effects.
    :param V: (n) or (n,v) numpy array of n observations of variables. This is the design matrix for the fixed variance effects.
    :param G: (n) or (n,l) numpy array of n observations of variables. This is the design matrix for the random effects.

    :return: heteroskedastic linear mixed model
    """
    def __init__(self,y,X,V,G):
        # Get sample size
        self.n = X.shape[0]
        # Check shape of arrays
        if G.ndim == 2:
            self.l = G.shape[1]
        elif G.ndim == 1:
            self.l = 1
            G = G.reshape((self.n, 1))
        else:
            raise (ValueError('Incorrect dimension of Random Effects Design Matrix'))
        if V.ndim == 2:
            self.n_fixed_variance = V.shape[1]
        elif V.ndim == 1:
            self.n_fixed_variance = 1
            V = V.reshape((self.n, 1))
        else:
            raise (ValueError('Incorrect dimension of Variance Covariate Array'))
        if X.ndim == 2:
            self.n_fixed_mean = X.shape[1]
        elif X.ndim == 1:
            self.n_fixed_mean = 1
            X = X.reshape((self.n, 1))
        else:
            raise (ValueError('Incorrect dimension of Mean Covariate Array'))
        # phenotype
        if y.ndim>1:
            raise(ValueError('Incorrect dimension of phenotype array'))
        self.y = y
        # mean covariates
        self.X = X
        # variance covariates
        self.V = V
        # random effects design matrix
        self.G = G

    def likelihood(self,beta,h2,negative=False):
        """
        Compute the log of the profile likelihood, the likelihood at the maximum likelihood for the fixed mean effects
        :param beta: value of fixed variance effects to compute likelihood for
        :param h2: value of variance explained by random effects to compute likelihood for
        :param negative: compute -2*L/n-log(2*pi), where L is the log-likelihood, the function that is minimized to find the MLE. Default is False.

        :return: L, log-likelihood of data given parameters.

        """

        L = self.likelihood_and_gradient(beta,h2,return_grad=False)
        if not negative:
            L = -0.5*self.n*(L+np.log(2*np.pi))
        return L

    # Compute likelihood of data given beta, alpha
    def likelihood_and_gradient(self,beta,h2,return_grad=True):
        """
                Compute the the function that is minimized to find the MLE, LL=-2*L/n-log(2*pi), where L is the log
                of the profile likelihood, the likelihood at the maximum for the fixed mean effects.
                Further, compute the gradient with respect to the fixed variance effects and the variance of the random effects.
                This forms the basis of the function passed to L-BFGS-B in order to find the maximum likelihood parameter estimates.
                :param beta: value of fixed variance effects
                :param h2: value of variance explained by random effects
                :param return_grad: whether to compute and return the gradient as well as the likelihood function. Default is True.

                :return: [LL,grad], the value of the function to be minimized, LL, and its gradient. The gradient is a 1d numpy array
                that has the gradient with respect to beta first followed by the gradient with respect to h2.
        """
        ## Calculate common variables
        # heteroscedasticity
        Vb = np.dot(self.V, beta)
        D_inv = np.exp(-Vb)
        # Low rank covariance
        G_scaled_T = self.G.T * D_inv
        G_scaled = G_scaled_T.T
        G_cov = np.dot(G_scaled_T, self.G)
        Lambda = np.identity(self.l, float) + h2 * G_cov
        Lambda = linalg.eigh(Lambda, overwrite_a=True, turbo=True)
        logdet_Lambda = np.sum(np.log(Lambda[0]))
        Lambda_inv = inv_from_eig(Lambda)
        ## Calculate MLE of fixed effects
        X_scaled = np.transpose(self.X.T * D_inv)
        alpha = alpha_mle_inner(h2, X_scaled, self.X, self.y, G_scaled, self.G, Lambda_inv)
        ## Residuals
        if self.n_fixed_mean > 1:
            resid = self.y - np.dot(self.X, alpha)
        else:
            yhat = self.X * alpha
            yhat = yhat.reshape(self.y.shape)
            resid = self.y - yhat
        ## Squared residuals
        resid_square = np.square(resid)
        rnd_resid = np.dot(G_scaled_T, resid)
        Lambda_inv_rnd_resid = np.dot(Lambda_inv, rnd_resid)
        ### Calculate likelihood
        L = np.sum(Vb) + np.sum(resid_square * D_inv) + logdet_Lambda - h2 * np.dot(np.transpose(rnd_resid),
                                                                                    Lambda_inv_rnd_resid)
        if return_grad:
            ### Calculate gradient
            grad = np.zeros((self.n_fixed_variance+1))
            # Calculate gradient with respect to beta
            k = var_weight(h2, resid, self.G, Lambda_inv, Lambda_inv_rnd_resid)
            n1t = np.ones((self.n)).reshape((1, self.n))
            grad[0:self.n_fixed_variance] = np.dot(n1t, np.transpose(np.transpose(self.V) * (1 - k * D_inv)))
            # Calculate gradient with respect to h2
            grad[self.n_fixed_variance] = grad_h2_inner(Lambda_inv, G_cov, Lambda_inv_rnd_resid)
            return L/np.float64(self.n), grad/np.float64(self.n)
        else:
            return L/np.float64(self.n)

    # OLS solution for alpha
    def alpha_ols(self):
        """
                Compute the ordinary least squares (OLS) estimate of the fixed mean effect parameters

                :return: alpha, 1d numpy array
        """
        # Get initial guess for alpha
        return np.linalg.solve(np.dot(self.X.T, self.X), np.dot(self.X.T, self.y))

    # Compute MLE of alpha given beta
    def alpha_mle(self,beta,h2):
        """
                Compute the maximum likelihood estimate of the fixed mean effect parameters, given
                particular fixed variance effect parameters and variance of random effects
                :param beta: value of fixed variance effects
                :param h2: value of variance explained by random effects

                :return: alpha, 1d numpy array
        """
        ## Calculate common variables
        # heteroscedasticity
        Vb = np.dot(self.V, beta)
        D_inv = np.exp(-Vb)
        # Low rank covariance
        G_scaled = np.transpose(self.G.T * D_inv)
        G_cov = np.dot(np.transpose(self.G), G_scaled)
        Lambda = np.identity(self.l, float) + h2 * G_cov
        Lambda_inv = linalg.inv(Lambda)
        ## Calculate MLE of fixed effects
        X_scaled = np.transpose(self.X.T * D_inv)
        return alpha_mle_inner(h2, X_scaled, self.X, self.y, G_scaled, self.G, Lambda_inv)



    def optimize_model(self,h2,SEs=True,dx=10**(-6)):
        """
                Find the maximum likelihood estimate (MLE) of the parameters and their sampling distribution.
                :param h2: initial value of variance explained by random effects
                :param SEs: whether to compute sampling distribution of parameter estimates. Default is True.
                :param dx: the step size used to compute the Hessian for the computing the parameter sampling distribution

                :return: optim, dictionary with MLEs ('alpha', fixed mean effects; 'beta', fixed variance effects;
                'h2', variance of random effects), their standard errors ('alpha_se', 'beta_se', 'h2_se'),
                covariance matrix for sampling distribution of parameter vector ('par_cov', in order: alpha, beta, h2),
                maximum likelihood ('likelihood'), whether optimisation was successful ('success'),
                warnings from L-BFGS-B optimisation ('warnflag').
        """
        # Initialise parameters
        init_params=np.zeros((self.n_fixed_variance+1))
        init_params[self.n_fixed_variance]=h2
        # Get initial guess for beta
        init_params[0:self.n_fixed_variance] = hetlm.model(self.y, self.X, self.V).optimize_model()['beta']
        ## Set parameter boundaries
        # boundaries for beta
        parbounds = [(None,None) for i in xrange(0,self.n_fixed_variance)]
        # boundaries for h2
        parbounds.append((0.00001, None))
        # Optimize
        optimized = fmin_l_bfgs_b(func=lik_and_grad_var_pars,x0=init_params,
                                args=(self.y, self.X, self.V, self.G),
                                bounds=parbounds)
        # Get MLE
        optim = {}
        optim['success']=True
        optim['warnflag'] = optimized[2]['warnflag']
        if optim['warnflag']!=0:
            print('Optimization unsuccessful.')
            optim['success']=False
        optim['beta'] = optimized[0][0:self.n_fixed_variance]
        optim['h2'] = optimized[0][self.n_fixed_variance]
        optim['alpha'] = self.alpha_mle(optim['beta'],optim['h2'])
        # Get parameter covariance
        optim['likelihood'] = -0.5*np.float64(self.n)*(optimized[1]+np.log(2*np.pi))

        # Compute parameter covariance
        if SEs:
            optim['par_cov'] = self.parameter_covariance(optim['alpha'],optim['beta'],optim['h2'],dx)
            par_se = np.sqrt(np.diag(optim['par_cov'] ))
            optim['alpha_se'] = par_se[0:self.n_fixed_mean]
            optim['beta_se'] = par_se[self.n_fixed_mean:(self.n_fixed_variance+self.n_fixed_mean)]
            optim['h2_se']=par_se[self.n_fixed_mean+self.n_fixed_variance]

        return optim

    # Calculate covariance and SEs of parameters at the MLE
    def parameter_covariance(self,alpha,beta,h2,dx=10**(-6)):
        ## Compute parameter covariance at maximum likelihood by numerical differentiation of gradients. Called from 'optimize_model'.
        # Calculate intermediate variables
        resid = (self.y - self.X.dot(alpha))
        # Residual Error
        D_inv = np.exp(-self.V.dot(beta))
        G_scaled_T = (self.G.T) * D_inv
        G_cov = G_scaled_T.dot(self.G)
        # Random Effect
        Lambda = np.identity(self.l, float) + h2 * G_scaled_T.dot(self.G)
        Lambda_inv = linalg.inv(Lambda, overwrite_a=True, check_finite=False)
        # Components of alpha gradient calculation
        X_scaled = np.transpose((self.X.T) * D_inv)
        X_grad_alpha = X_scaled - h2 * np.dot(np.dot(G_scaled_T.T, Lambda_inv), G_scaled_T.dot(self.X))
        # Form Hessian matrix
        n_pars=self.n_fixed_mean+self.n_fixed_variance+1
        H = np.zeros((n_pars, n_pars))
        # Calculate alpha components of hessian
        for p in xrange(0, self.n_fixed_mean):
            # Calculate change in alpha gradient
            d = np.identity(self.n_fixed_mean)*dx
            resid_upper = (self.y - self.X.dot(alpha + d[p,:]))
            resid_lower = (self.y - self.X.dot(alpha - d[p,:]))
            H[0:self.n_fixed_mean, p] = (grad_alpha(resid_upper, X_grad_alpha) - grad_alpha(resid_lower, X_grad_alpha)) / (
            2.0 * dx)
            # Calculate change in beta gradient
            H[self.n_fixed_mean:(n_pars - 1), p] = (grad_beta(h2, self.G, G_scaled_T, self.V, D_inv, resid_upper,
                                                         Lambda_inv) - grad_beta(h2, self.G, G_scaled_T, self.V, D_inv,
                                                                                 resid_lower, Lambda_inv)) / (2.0 * dx)
            H[p, self.n_fixed_mean:(n_pars - 1)] = H[self.n_fixed_mean:(self.n_fixed_mean + self.n_fixed_variance), p]
            # Calculate change in h2 gradient
            H[n_pars - 1, p] = (grad_h2_parcov(G_scaled_T, G_cov, resid_upper, Lambda_inv) - grad_h2_parcov(G_scaled_T, G_cov,
                                                                                              resid_lower,
                                                                                              Lambda_inv)) / (2.0 * dx)
            H[p, n_pars - 1] = H[n_pars - 1, p]
        # Calculate beta components of Hessian
        for p in xrange(self.n_fixed_mean, n_pars - 1):
            d = np.identity(self.n_fixed_variance) * dx
            # Changed matrices
            D_inv_upper = np.exp(-self.V.dot(beta + d[p-self.n_fixed_mean,:]))
            D_inv_lower = np.exp(-self.V.dot(beta - d[p-self.n_fixed_mean,:]))
            G_scaled_T_upper = (self.G.T) * D_inv_upper
            G_scaled_T_lower = (self.G.T) * D_inv_lower
            G_cov_upper = np.dot(G_scaled_T_upper, self.G)
            G_cov_lower = np.dot(G_scaled_T_lower, self.G)
            Lambda_inv_upper = linalg.inv(np.identity(self.l, float) + h2 * G_cov_upper, overwrite_a=True,
                                          check_finite=False)
            Lambda_inv_lower = linalg.inv(np.identity(self.l, float) + h2 * G_cov_lower, overwrite_a=True,
                                          check_finite=False)
            # Change in beta gradient
            H[self.n_fixed_mean:(n_pars - 1), p] = (grad_beta(h2, self.G, G_scaled_T_upper, self.V, D_inv_upper, resid,
                                                         Lambda_inv_upper) - grad_beta(h2, self.G, G_scaled_T_lower, self.V,
                                                                                       D_inv_lower, resid,
                                                                                       Lambda_inv_lower)) / (2.0 * dx)
            # Change in h2 gradient
            H[n_pars - 1, p] = (grad_h2_parcov(G_scaled_T_upper, G_cov_upper, resid, Lambda_inv_upper) - grad_h2_parcov(
                G_scaled_T_lower, G_cov_lower, resid, Lambda_inv_lower)) / (2.0 * dx)
            H[p, n_pars - 1] = H[n_pars - 1, p]
        # Calculate h2 components of the Hessian
        Lambda_inv_upper = linalg.inv(np.identity(self.l, float) + (h2 + dx) * G_cov, overwrite_a=True, check_finite=False)
        Lambda_inv_lower = linalg.inv(np.identity(self.l, float) + (h2 - dx) * G_cov, overwrite_a=True, check_finite=False)
        H[n_pars - 1, n_pars - 1] = (grad_h2_parcov(G_scaled_T, G_cov, resid, Lambda_inv_upper) - grad_h2_parcov(G_scaled_T, G_cov,
                                                                                                   resid,
                                                                                                   Lambda_inv_lower)) / (
                                    2.0 * dx)
        par_cov = linalg.inv(0.5 * H, overwrite_a=True, check_finite=False)
        return par_cov



def simulate(n,l,alpha,beta,h2):
    """
            Simulate from a heteroskedastic linear mixed model given a set of parameters. This uses a singular
            value decomposition to do the simulation quickly when l<<n.

            The function simulates fixed and random effects design matrices of specified dimensions with independent Gaussian entries.

            :param n: sample size
            :param l: number of random effects
            :param alpha: value of fixed mean effects
            :param beta: value of fixed variance effects
            :param h2: value of variance explained by random effects

            :return: hetlmm.model class with simulated data and phenotypes for the given parameters.
    """
    if (l>n):
        print('Simulation slow for l>n')
    c = alpha.shape[0]
    v = beta.shape[0]
    X = np.random.randn((n * c)).reshape((n, c))
    V = np.random.randn((n * v)).reshape((n, v))
    G = zscore(np.random.binomial(2,0.5,(n,l)),axis=0)*np.power(l,-0.5)
    G_svd = np.linalg.svd(G,full_matrices=False)
    y = np.sqrt(h2)*G_svd[0].dot(np.random.randn((l))*G_svd[1])+np.random.randn((n))*np.exp(V.dot(beta)/2.0)+X.dot(alpha)
    return model(y,X,V,G)

def lik_and_grad_var_pars(pars,*args):
    # Wrapper for function to pass to L-BFGS-B
    y, X, V, G = args
    hlmm_mod = model(y,X,V,G)
    return hlmm_mod.likelihood_and_gradient(pars[0:hlmm_mod.n_fixed_variance],pars[hlmm_mod.n_fixed_variance])

# Only for positive semi-definite matrix
def inv_from_eig(eig):
    # Compute matrix inverse from eigendecomposition
    U_scaled=eig[1]*np.power(eig[0],-0.5)
    return np.dot(U_scaled,U_scaled.T)


def alpha_mle_inner(h2,X_scaled,X,y,Z_scaled,Z,Lambda_inv):
    # Compute the maximum likelihood estimate of the fixed mean effects from intermediate variables in likelihood_and_gradient
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

####### Functions for efficient gradient computation ########

def var_weight(h2,resid,G,Lambda_inv,Lambda_inv_rnd_resid):
    # Compute the the weights needed for computation of the gradient with respect to beta
    cov_diagonal=(G.dot(Lambda_inv) * G).sum(-1)
    # Compute weights coming from inner product in low rank space
    a=G.dot(Lambda_inv_rnd_resid)
    k=np.square(resid)+h2*cov_diagonal+h2*a*(h2*a-2*resid)
    return k

def grad_beta(h2,G,G_scaled_T,V,D_inv,resid,Lambda_inv):
    # Compute gradient with respect to variance of random effects from intermediate variables in parameter_covariance
    n=V.shape[0]
    # Low rank covariance
    rnd_resid=np.dot(G_scaled_T,resid)
    Lambda_inv_rnd_resid=np.dot(Lambda_inv,rnd_resid)
    ### Calculate likelihood
    # Get k variance weights
    k=var_weight(h2,resid,G,Lambda_inv,Lambda_inv_rnd_resid)
    n1t=np.ones((n)).reshape((1,n))
    return np.dot(n1t,np.transpose(np.transpose(V)*(1-k*D_inv)))

def grad_h2_inner(Lambda_inv,Z_cov,Lambda_inv_rnd_resid):
    # Compute gradient with respect to variance of random effects from intermediate variables in likelihood_and_gradient
    dl=0
    # Calculate trace
    dl+=np.sum(Lambda_inv*Z_cov)
    # Calculate inner product
    dl+=-np.sum(np.square(Lambda_inv_rnd_resid))
    return dl

####### Functions for efficient computation of parameter covariance ########
def grad_h2_parcov(G_scaled_T,G_cov,resid,Lambda_inv):
    # Compute gradient with respect to variance of random effects from intermediate variables in parameter_covariance
    rnd_resid=G_scaled_T.dot(resid)
    Lambda_inv_rnd_resid=Lambda_inv.dot(rnd_resid)
    return grad_h2_inner(Lambda_inv,G_cov,Lambda_inv_rnd_resid)


def grad_alpha(resid,X_grad_alpha):
    # Compute gradient with respect to alpha from intermediate variables in parameter_covariance
    return -2*np.dot(resid.T,X_grad_alpha)