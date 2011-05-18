import numpy as np
from numpy import pi
from sundials import CVodeSolver

from .legendre import compute_normalized_associated_legendre

__all__ = ['associated_legendre_roots']

class AssociatedLegendreRootFinder:
    def __init__(self, l, m):
        self.l = l
        self.m = m

    def _rhs(self, theta, y):
        """ ODE rhs"""
        l, m = self.l, self.m
        x = y[0]
        if abs(x) >= 1:
            raise Exception()
        
        q = l * (l + 1) - m**2 / (1 - x**2)
        p = 1 - x**2
        dq = -2 * x * m**2 / (1 - x**2)**2
        dp = -2 * x
        v = np.sqrt(q / p) + 0.25 * np.sin(2 * theta) * (dq / q + dp / p)
        dx_dtheta = -1 / v
        assert type(dx_dtheta) == np.longdouble
        return [dx_dtheta]

    def find_roots(self):
        l, m = self.l, self.m
        solver = CVodeSolver(RHS=self._rhs,
                             abstol=1e-19,
                             reltol=1e-19,
                             mxsteps=2000)

        # Initial conditions
        even = (l - m) % 2 == 0
        root = 0 # Start in x(theta) = 0
        if even:
            # For even polynomials, P_lm(0) is a maxima/minima,
            # corresponding to theta=0
            theta0 = 0            
        else:
            # For odd polynomials, P_lm(0) is a root
            theta0 = .5 * pi
        numroots = (l - m) // 2
        roots = np.zeros(numroots)
        for i in range(numroots):
            solver.init(theta0, np.asarray([root], np.float128))
            root = roots[i] = solver.step(-.5 * pi)[0]
            theta0 = .5 * pi # in case of even above
        return roots
        

def associated_legendre_roots(l, m):
    """ Returns the root of P_lm(x) in the interval (0, 1)
    """
    return AssociatedLegendreRootFinder(l, m).find_roots()
