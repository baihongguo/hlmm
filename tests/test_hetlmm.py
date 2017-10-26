import unittest

import numdifftools as nd
import numpy as np
from numpy import testing

from hlmm import hetlmm


class test_hlmm_functions(unittest.TestCase):



    def test_likelihood_and_beta(self):
        for n in [10**2]:
            for v in [2]:
                if v>n:
                    v=n
                for c in [2]:
                    if c>n:
                        c=n
                    for i in xrange(0,100):
                        l=100
                        alpha=np.random.randn((c))
                        beta = np.random.randn((v))/10
                        model = hetlmm.simulate(n, l, alpha, beta, 0.1)
                        Vb = np.dot(model.V, beta)
                        Sigma = np.diag(np.exp(Vb))+0.1*model.G.dot(model.G.T)
                        logdet=np.linalg.slogdet(Sigma)
                        logdet=logdet[0]*logdet[1]
                        Sigma_inv = np.linalg.inv(Sigma)
                        L = model.likelihood(beta,0.1)
                        resid=model.y-model.X.dot(model.alpha_mle(beta,0.1))
                        safe_lik=np.dot(resid.T,Sigma_inv.dot(resid))+logdet
                        testing.assert_almost_equal(L,safe_lik/float(n),decimal=5)


    def test_alpha_mle(self):
        for n in [10**2]:
            for v in [2]:
                if v>n:
                    v=n
                for c in [2]:
                    if c>n:
                        c=n
                    for i in xrange(0,100):
                        l = 100
                        alpha = np.random.randn((c))
                        beta = np.random.randn((v)) / 10
                        model = hetlmm.simulate(n, l, alpha, beta, 0.1)
                        Vb = np.dot(model.V, beta)
                        Sigma = np.diag(np.exp(Vb)) + 0.1 * model.G.dot(model.G.T)
                        Sigma_inv=np.linalg.inv(Sigma)
                        safe_alpha=np.linalg.solve(np.dot(model.X.T,Sigma_inv.dot(model.X)),np.dot(model.X.T,Sigma_inv.dot(model.y)))
                        testing.assert_almost_equal(model.alpha_mle(beta,0.1),safe_alpha,decimal=5)

    def test_grad_beta(self):
        for n in [10**2]:
            for v in [2,10,100]:
                if v>n:
                    v=n
                for c in [2]:
                    if c>n:
                        c=n
                    for i in xrange(0,100):
                        l = 100
                        alpha = np.random.randn((c))/10
                        beta = np.random.randn((v)) / 10
                        model = hetlmm.simulate(n, l, alpha, beta, 0.1)
                        vpar=np.zeros((v+1))
                        vpar[0:v]=np.random.randn((v))/10
                        vpar[v]=np.random.uniform(0.0005,1)
                        L, gradb = model.likelihood_and_gradient(vpar[0:v], vpar[v])
                        # Compute gradient numerically
                        def likelihood(vpars):
                            return model.likelihood(vpars[0:v],vpars[v])
                        num_grad=np.zeros((v+1))
                        diffs=np.identity(v+1)*10**(-6)
                        for i in xrange(0,v+1):
                            num_grad[i]=(likelihood(vpar+diffs[i,:])-likelihood(vpar-diffs[i,:]))/(2*10**(-6))
                        # Compute analytically
                        testing.assert_almost_equal(num_grad,gradb.reshape((v+1)),decimal=5)



if  __name__=='__main__':
	unittest.main()