from hlmm import hetlm
import numdifftools as nd
import numpy as np
import unittest
from numpy import testing


class test_hetlm_functions(unittest.TestCase):

    def test_likelihood(self):
        for n in [1,2,10**2]:
            for v in [1,2,10]:
                if v>n:
                    v=n
                for c in [1,2,10]:
                    if c>n:
                        c=n
                    for i in xrange(0,10**3):
                        X=np.random.randn((n*c)).reshape((n,c))
                        V = np.random.randn((n * v)).reshape((n, v))
                        y=np.random.randn((n))
                        alpha=np.random.randn((c))
                        beta = np.random.randn((v))
                        Vb = np.dot(V, beta)
                        Sigma = np.diag(np.exp(Vb))
                        logdet=np.linalg.slogdet(Sigma)
                        logdet=logdet[0]*logdet[1]
                        Sigma_inv = np.linalg.inv(Sigma)
                        hetlm_mod= hetlm.model(y, X, V)
                        lik=hetlm_mod.likelihood(beta,alpha)
                        resid=y-X.dot(alpha)
                        safe_lik=np.dot(resid.T,Sigma_inv.dot(resid))+logdet
                        testing.assert_almost_equal(lik,safe_lik,decimal=5)

    def test_alpha_mle(self):
        for n in [1,2,10**2]:
            for v in [1,2,10]:
                if v>n:
                    v=n
                for c in [1,2,10]:
                    if c>n:
                        c=n
                    for i in xrange(0,10**3):
                        X=np.random.randn((n*c)).reshape((n,c))
                        V=np.random.randn((n*v)).reshape((n,v))
                        alpha=np.random.randn((c))
                        y=np.random.randn((n))
                        beta=np.random.randn((v))/10
                        Vb = np.dot(V, beta)
                        y=y*np.exp(Vb/2.0)+X.dot(alpha)
                        Sigma = np.diag(np.exp(Vb))
                        Sigma_inv=np.linalg.inv(Sigma)
                        hetlm_mod= hetlm.model(y, X, V)
                        alpha=hetlm_mod.alpha_mle(beta)
                        safe_alpha=np.linalg.solve(np.dot(X.T,Sigma_inv.dot(X)),np.dot(X.T,Sigma_inv.dot(y)))
                        testing.assert_almost_equal(alpha,safe_alpha,decimal=5)

    def test_grad_beta(self):
        for n in [1,2,10**2]:
            for v in [1,2,10]:
                if v>n:
                    v=n
                for c in [1,2,10]:
                    if c>n:
                        c=n
                    for i in xrange(0,10**3):
                        X=np.random.randn((n*c)).reshape((n,c))
                        V = np.random.randn((n * v)).reshape((n, v))
                        y=np.random.randn((n))
                        hetlm_mod = hetlm.model(y, X, V)
                        alpha=np.zeros((c))
                        def likelihood(beta):
                            return hetlm_mod.likelihood(beta,alpha)
                        # Compute gradient numerically
                        num_grad=nd.Gradient(likelihood)(np.zeros((v)))
                        testing.assert_almost_equal(num_grad,hlm_mod.grad_beta(np.zeros((v)),alpha).reshape((v)),decimal=5)



if  __name__=='__main__':
	unittest.main()