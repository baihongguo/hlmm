import numpy as np
from scipy.optimize import minimize

class model(object):
    """
    A heteroskedastic linear model
    """
    def __init__(self,y,X,V):
        # Get sample size
        self.n = X.shape[0]
        # Check shape of arrays
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
        self.y=y
        # mean covariates
        self.X=X
        # variance covariates
        self.V=V

    # Compute likelihood of data given beta, alpha
    def likelihood(self,beta,alpha):
        Vbeta = self.V.dot(beta)
        resid = self.y - self.X.dot(alpha)
        L = np.sum(Vbeta) + np.sum(np.square(resid) * np.exp(-Vbeta))
        # print('Likelihood: '+str(round(L,5)))
        return L

    # Compute MLE of alpha given beta
    def alpha_mle(self,beta):
        D_inv = np.exp(-self.V.dot(beta))
        X_t_D_inv = np.transpose(self.X) * D_inv
        alpha = np.linalg.solve(X_t_D_inv.dot(self.X), X_t_D_inv.dot(self.y))
        return alpha

    # Compute gradient with respect to beta for a given beta and alpha
    def grad_beta(self,beta, alpha):
        D_inv = np.exp(-self.V.dot(beta))
        resid_2 = np.square(self.y - self.X.dot(alpha))
        k = 1 - resid_2 * D_inv
        V_scaled = np.transpose(np.transpose(self.V) * k)
        n1t = np.ones((1, self.X.shape[0]))
        return n1t.dot(V_scaled)

    # OLS solution for alpha
    def alpha_ols(self):
        # Get initial guess for alpha
        return np.linalg.solve(np.dot(self.X.T, self.X), np.dot(self.X.T, self.y))

    # Find an approximation to the MLE of beta given alpha
    def approx_beta_mle(self):
        # Get alpha OLS
        alpha=self.alpha_ols()
        # squared residuals
        resid_2=np.square(self.y-self.X.dot(alpha)).reshape((self.X.shape[0]))
        # RHS
        V_scaled=np.transpose(np.transpose(self.V)*(resid_2-1))
        n1t=np.ones((1,self.X.shape[0]))
        b=n1t.dot(V_scaled).reshape(self.V.shape[1])
        # LHS
        V_t_scaled=np.transpose(self.V)*resid_2
        A=V_t_scaled.dot(self.V)
        return np.linalg.solve(A,b)

    # Find the covariance matrix of beta
    def beta_cov(self):
        return 2 * np.linalg.inv(np.dot(self.V.T, self.V))

    # Find the covariance matrix for alpha given beta
    def alpha_cov(self,beta):
        D_inv=np.exp(-self.V.dot(beta))
        precision=np.dot(np.transpose(self.X)*D_inv,self.X)
        return np.linalg.inv(precision)

    def optimize_model(self):
        # Get initial guess for beta
        beta_init = self.approx_beta_mle()
        # Optimize
        optimized = minimize(likelihood_beta, beta_init,
                             args=(self.y, self.X, self.V),
                             method='L-BFGS-B',
                             jac=gradient_beta)
        if not optimized.success:
            print('Optimization unsuccessful.')
        # Get MLE
        beta_mle = optimized['x']
        alpha = self.alpha_mle(beta_mle)
        # Get parameter covariance
        optim = {}
        optim['beta'] = optimized['x']
        optim['alpha'] = self.alpha_mle(optim['beta'])
        optim['beta_cov'] = self.beta_cov()
        optim['beta_se'] = np.sqrt(np.diag(optim['beta_cov']))
        optim['alpha_cov'] = self.alpha_cov(beta_mle)
        optim['alpha_se'] = np.sqrt(np.diag(optim['alpha_cov']))
        optim['likelihood'] = -0.5 * (optimized['fun'] + self.n * np.log(2 * np.pi))
        optim['success']=optimized.success
        return optim

##### Functions to pass to opimizer ######
# Profile likelihood of hlm_model as a function of beta
def likelihood_beta(beta,*args):
    y,X,V=args
    n=np.float(X.shape[0])
    hlm_mod=model(y,X,V)
    alpha=hlm_mod.alpha_mle(beta)
    return hlm_mod.likelihood(beta,alpha)/n


# Gradient of likelihood with respect to beta at the MLE of alpha
def gradient_beta(beta, *args):
    y,X,V=args
    n = np.float(X.shape[0])
    hlm_mod=model(y,X,V)
    alpha = hlm_mod.alpha_mle(beta)
    return hlm_mod.grad_beta(beta, alpha).reshape((hlm_mod.V.shape[1]))/n


