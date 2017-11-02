import numpy as np
from scipy.optimize import minimize

class model(object):
    """
    Define a heteroskedastic linear model and calculate likelihood, gradients, and maximum likelihood estimates of
    parameters.
    :param y: (n) or (n,1) numpy array of phenotype observations
    :param X: (n) or (n,c) numpy array of n observations of variables. This is the design matrix for the fixed mean effects.
    :param V: (n) or (n,v) numpy array of n observations of variables. This is the design matrix for the fixed variance effects.

    :return: heteroskedastic linear model
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
    def likelihood(self,beta,alpha,negative=False):
        """
        Compute the log of the likelihood, the likelihood at the maximum likelihood for the fixed mean effects
        :param alpha: value of fixed mean effects to compute likelihood for
        :param beta: value of fixed variance effects to compute likelihood for
        :param negative: compute -2*L/n-log(2*pi), where L is the log-likelihood, the function that is minimized to find the MLE. Default is False.

        :return: L, log-likelihood of data given parameters.

        """
        Vbeta = self.V.dot(beta)
        resid = self.y - self.X.dot(alpha)
        L = np.sum(Vbeta) + np.sum(np.square(resid) * np.exp(-Vbeta))
        if not negative:
            L = -0.5*(L+self.n*np.log(2*np.pi))
        return L

    # Compute MLE of alpha given beta
    def alpha_mle(self,beta):
        """
                Compute the maximum likelihood estimate of the fixed mean effect parameters, given
                particular fixed variance effect parameters and variance of random effects
                :param beta: value of fixed variance effects


                :return: alpha, 1d numpy array
        """
        D_inv = np.exp(-self.V.dot(beta))
        X_t_D_inv = np.transpose(self.X) * D_inv
        alpha = np.linalg.solve(X_t_D_inv.dot(self.X), X_t_D_inv.dot(self.y))
        return alpha

    # Compute gradient with respect to beta for a given beta and alpha
    def grad_beta(self,beta, alpha):
        """
        Compute the gradient with respect to the fixed variance effects of -2*L/n-log(2*pi),
        where L is the log-likelihood, the function that is minimized to find the MLE
        :param alpha: value of fixed mean effects to compute likelihood for
        :param beta: value of fixed variance effects to compute likelihood for

        :return: grad_beta, 1d numpy array

        """
        D_inv = np.exp(-self.V.dot(beta))
        resid_2 = np.square(self.y - self.X.dot(alpha))
        k = 1 - resid_2 * D_inv
        V_scaled = np.transpose(np.transpose(self.V) * k)
        n1t = np.ones((1, self.X.shape[0]))
        return n1t.dot(V_scaled)

    # OLS solution for alpha
    def alpha_ols(self):
        """
                Compute the ordinary least squares (OLS) estimate of the fixed mean effect parameters

                :return: alpha, 1d numpy array
        """
        # Get initial guess for alpha
        return np.linalg.solve(np.dot(self.X.T, self.X), np.dot(self.X.T, self.y))

    # Find an approximation to the MLE of beta given alpha
    def approx_beta_mle(self):
        """
                Analytical approximation to the maximum likelihood estimate of the fixed variance effects

                :return: beta, 1d numpy array
        """
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

    def beta_cov(self):
        # Find the covariance matrix of beta
        return 2 * np.linalg.inv(np.dot(self.V.T, self.V))

    def alpha_cov(self,beta):
        # Find the covariance matrix for alpha given beta
        D_inv=np.exp(-self.V.dot(beta))
        precision=np.dot(np.transpose(self.X)*D_inv,self.X)
        return np.linalg.inv(precision)

    def optimize_model(self):
        """
                Find the maximum likelihood estimate (MLE) of the parameters and their sampling distribution.

                :return: optim, dictionary with MLEs ('alpha', fixed mean effects; 'beta', fixed variance effects),
                their standard errors ('alpha_se', 'beta_se', 'h2_se'),
                covariance matrix for sampling distribution of parameter vectors ('beta_cov' and 'alpha_cov'),
                maximum likelihood ('likelihood'), whether optimisation was successful ('success'),
        """
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
        optim['likelihood'] = -0.5 *self.n * (optimized['fun'] + np.log(2 * np.pi))
        optim['success']=optimized.success
        return optim

##### Functions to pass to opimizer ######
def likelihood_beta(beta,*args):
    # Profile likelihood of hlm_model as a function of beta. To pass to L-BFGS-B
    y,X,V=args
    n=np.float(X.shape[0])
    hlm_mod=model(y,X,V)
    alpha=hlm_mod.alpha_mle(beta)
    return hlm_mod.likelihood(beta,alpha,negative=True)/n


def gradient_beta(beta, *args):
    # Gradient of profile likelihood with respect to beta (at the MLE of alpha). To pass to L-BFGS-B.
    y,X,V=args
    n = np.float(X.shape[0])
    hlm_mod=model(y,X,V)
    alpha = hlm_mod.alpha_mle(beta)
    return hlm_mod.grad_beta(beta, alpha).reshape((hlm_mod.V.shape[1]))/n


