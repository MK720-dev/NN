import numpy as np 
from  scipy.optimize import minimize # type: ignore


# ---------------------
# Line Search Helpers
#----------------------

#---- Helper: Backtracking Line Search using Armijo Condition ----
def backtracking_line_search(theta, p, grad, cost_fn, alpha_init=1.0, rho=0.5, c=1e-4):
    alpha = alpha_init 
    f0 = cost_fn(theta)
    while cost_fn(theta + alpha * p) > f0 + c * alpha * (grad @ p):
        alpha *= rho
    return alpha

#---- Helper: A more robust Armijo Backtracking Line Search with upper and lower bounds on shrinkage and safeguards ----
def armijo_backtracking_interp(x, p, f, grad, alpha=1e-4, l=0.1, u=0.5, max_iter=20):
    """
    Armijo backtracking line search with quadratic/cubic interpolation.
    
    Parameters
    ----------
    x : np.ndarray
        Current point.
    p : np.ndarray
        Search direction (should be descent).
    f : callable
        Function f(theta).
    grad : callable
        Function grad(theta).
    alpha : float
        Armijo parameter (default 1e-4).
    l, u : float
        Lower/upper bounds for shrinkage factor rho.
    max_iter : int
        Maximum backtracking iterations.

    Returns
    -------
    lam : float
        Step length satisfying Armijo condition.
    """

    f0 = f(x)
    g0 = grad(x)
    d0 = float(g0 @ p)
    if d0 >= 0:
        raise ValueError("Search direction is not descent")

    lam = 1.0
    f_prev, lam_prev = None, None

    for it in range(max_iter):
        f_trial = f(x + lam * p)

        # Check Armijo condition
        if f_trial <= f0 + alpha * lam * d0:
            return lam

        # First backtrack: quadratic interpolation
        if f_prev is None:
            denom = 2 * (f_trial - f0 - lam * d0)
            if denom <= 0:  # fallback
                lam_new = lam * u
            else:
                lam_new = -d0 * lam**2 / denom
            lam_new = np.clip(lam_new, l*lam, u*lam)

        # Subsequent backtracks: cubic interpolation
        else:
            f_pprev, lam_pprev = f_prev
            A = np.array([
                [1/lam_prev**2, -1/lam_pprev**2],
                [-lam_prev,      lam_pprev]
            ])
            rhs = np.array([
                (f_trial - f0 - lam_prev*d0) / lam_prev**2,
                (f_pprev - f0 - lam_pprev*d0) / lam_pprev**2
            ])
            coeffs = np.linalg.solve(A, rhs)
            a, b = coeffs
            disc = b**2 - 3*a*d0
            if a == 0 or disc < 0:
                lam_new = lam * u
            else:
                lam_new = (-b + np.sqrt(disc)) / (3*a)
            lam_new = np.clip(lam_new, l*lam, u*lam)

        # Update history
        f_prev = (f_trial, lam)
        lam_prev = lam
        lam = lam_new

    return lam  # fallback if max_iter reached


#---- Helper: Backtracking Line Search using Weak Wolfe Conditions ----
def backtracking_weak_wolfe(x, p, f, grad,
                            c1=1e-4, c2=0.9,
                            lam_init=1.0, lam_max=10.0, lam_min=1e-8,
                            rho=0.5, max_iter=20):
    """
    Backtracking line search with weak Wolfe conditions (faithful to A6.3.1mod).
    """

    f0 = f(x)
    g0 = grad(x)
    d0 = float(g0 @ p)
    if d0 >= 0:
        raise ValueError("p is not a descent direction")

    lam = lam_init
    lam_lo, lam_hi = None, None
    f_lo, f_hi = None, None

    first_backtrack = True

    for _ in range(max_iter):
        f_trial = f(x + lam * p)
        g_trial = grad(x + lam * p)
        d_trial = float(g_trial @ p)

        # Wolfe conditions
        if f_trial <= f0 + c1 * lam * d0 and d_trial >= c2 * d0:
            return lam

        # Step 4: Armijo holds and λ<1 OR Armijo fails and λ>1
        if (f_trial <= f0 + c1 * lam * d0 and lam < 1) or (f_trial > f0 + c1 * lam * d0 and lam > 1):
            if f_trial <= f0 + c1 * lam * d0:  # Armijo true, curvature fails
                lam_lo, f_lo = lam, f_trial
            else:  # Armijo fails
                lam_hi, f_hi = lam, f_trial

            if lam_lo is not None and lam_hi is not None:
                # Quadratic interpolation inside [lo,hi]
                num = (f_hi - f0 - lam_hi * d0) * lam_lo**2 - (f_lo - f0 - lam_lo * d0) * lam_hi**2
                den = (f_hi - f0 - lam_hi * d0) * lam_lo - (f_lo - f0 - lam_lo * d0) * lam_hi
                if den != 0:
                    lam_new = num / (2 * den)
                else:
                    lam_new = 0.5 * (lam_lo + lam_hi)
                lam = np.clip(lam_new, lam_min, lam_max)
            else:
                lam *= rho
            continue

        # Step 5: Armijo fails and λ ≤ 1
        if f_trial > f0 + c1 * lam * d0 and lam <= 1:
            if first_backtrack:
                # Quadratic interpolation
                denom = 2 * (f_trial - f0 - lam * d0)
                if denom > 0:
                    lam_new = -d0 * lam**2 / denom
                else:
                    lam_new = rho * lam
                lam = np.clip(lam_new, lam_min, lam_max)
                first_backtrack = False
            else:
                # Cubic interpolation using previous values
                if lam_lo is not None:
                    num = (lam**2) * (f_lo - f0 - lam_lo * d0) - (lam_lo**2) * (f_trial - f0 - lam * d0)
                    den = (lam**3) * (f_lo - f0 - lam_lo * d0) - (lam_lo**3) * (f_trial - f0 - lam * d0)
                    if den != 0:
                        lam_new = num / (3 * den)
                    else:
                        lam_new = rho * lam
                    lam = np.clip(lam_new, lam_min, lam_max)
                else:
                    lam *= rho
            continue

        # If curvature fails but Armijo holds and λ ≥ 1 → expand
        if f_trial <= f0 + c1 * lam * d0 and d_trial < c2 * d0 and lam >= 1:
            lam = min(2 * lam, lam_max)
            continue

        if lam < lam_min:
            break

    return lam
"""
def backtracking_weak_wolfe(x, p, f, grad, alpha=1e-4, beta=0.9, l=0.1, u=0.5, max_iter=20):
    
    Backtracking line search satisfying weak Wolfe conditions.
    
    f0 = f(x)
    g0 = grad(x)
    d0 = float(g0 @ p)   # slope at 0 (should be < 0)
    if d0 >= 0:
        raise ValueError("p is not a descent direction")

    lam = 1.0
    f_prev, lam_prev = None, None

    for it in range(max_iter):
        f_trial = f(x + lam * p)
        g_trial = grad(x + lam * p)
        d_trial = float(g_trial @ p)

        # Wolfe conditions
        if (f_trial <= f0 + alpha * lam * d0) and (d_trial >= beta * d0):
            return lam

        # Case A: Armijo holds but curvature fails (too steep)
        if f_trial <= f0 + alpha * lam * d0 and d_trial < beta * d0:
            # try increasing step if lam=1
            lam *= 2.0
            continue

        # Case B: Armijo fails
        if f_prev is None:
            # quadratic interpolation
            denom = 2 * (f_trial - f0 - lam * d0)
            if denom <= 0:
                lam_new = lam * u
            else:
                lam_new = -d0 * lam**2 / denom
            lam_new = np.clip(lam_new, l*lam, u*lam)
        else:
            # cubic interpolation
            f_pprev, lam_pprev = f_prev
            A = np.array([
                [1/lam**2, -1/lam_pprev**2],
                [-lam, lam_pprev]
            ])
            rhs = np.array([
                (f_trial - f0 - lam*d0) / lam**2,
                (f_pprev - f0 - lam_pprev*d0) / lam_pprev**2
            ])
            coeffs = np.linalg.solve(A, rhs)
            a, b = coeffs
            disc = b**2 - 3*a*d0
            if a == 0 or disc < 0:
                lam_new = lam * u
            else:
                lam_new = (-b + np.sqrt(disc)) / (3*a)
            lam_new = np.clip(lam_new, l*lam, u*lam)

        # update history
        f_prev = (f_trial, lam)
        lam_prev = lam
        lam = lam_new

    return lam  # fallback
"""

#----------------------
# Base Class
#----------------------
class Optimizer:
    def step(self, theta, grad, cost_fn, grad_fn=None):
        raise NotImplementedError

#-----------------------------------
# Batch Gradient Descent (fixed lr)
#-----------------------------------
class BatchGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, theta, grad, cost_fn=None, grad_fn=None):
        return theta - self.lr * grad

#------------------------------------------------------------
# Gradient Descent + Backtracking Line Search using Armijo
#------------------------------------------------------------
class LineSearchGD(Optimizer):
    def __init__(self, lambda_init=1.0, rho=0.5, c=1e-4):
        self.lambda_init, self.rho, self.c = lambda_init, rho, c

    def step(self, theta, grad, cost_fn, grad_fn=None):
        p = -grad
        alpha = armijo_backtracking_interp(theta, p, cost_fn, grad_fn)
        return theta + alpha * p


#-------------------------------------------------------------
# BFGS + Backtracking Line Search using Weak Wolfe conditions
#-------------------------------------------------------------
class BFGS_Wolfe(Optimizer):
    def __init__(self, alpha=1e-4, beta=0.9, max_iter=50):
        self.H = None  # Hessian approximation
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter

    def step(self, theta, grad, cost_fn, grad_fn):
        """
        Perform one BFGS update with weak Wolfe line search.
        
        Parameters
        ----------
        theta : np.ndarray
            Current parameter vector.
        grad : np.ndarray
            Current gradient at theta.
        cost_fn : callable
            Function f(theta).
        grad_fn : callable
            Function grad(theta).

        Returns
        -------
        theta_new, grad_new : updated parameters and gradient
        """

        n = theta.size
        if self.H is None:
            self.H = np.eye(n)

        # Compute search direction
        p = -self.H @ grad

        # Line search with weak Wolfe conditions
        lam = backtracking_weak_wolfe(
            theta, p, cost_fn, grad_fn,
            c1=self.alpha, c2=self.beta, lam_min=1e-1, max_iter=self.max_iter
        )

        # Update theta
        theta_new = theta + lam * p
        grad_new = grad_fn(theta_new)

        # BFGS update
        s = theta_new - theta
        y = grad_new - grad
        ys = float(y @ s)

        if ys > 1e-10:  # safeguard against division by zero
            rho = 1.0 / ys
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            self.H = V @ self.H @ V.T + rho * np.outer(s, s)

        return theta_new, grad_new

#---------------------------------------------------------------------
# Scipy's BFGS implementation
#---------------------------------------------------------------------
class ScipyBFGS(Optimizer):
    def step(self, theta, grad, cost_fn, grad_fn):
        res = minimize(cost_fn, theta, jac=grad_fn, method="BFGS",
                       options={"maxiter": 1})  # step at a time
        return res.x, grad_fn(res.x)

#---------------------------------------------------------------------
# SGD with Armijo Line Search
#---------------------------------------------------------------------
class LineSearchSGD(Optimizer):
    def step(self, theta, grad, loss_fn, grad_fn, x, y):
        """
        One SGD update with Armijo line search.
        """
        p = -grad

        # wrap loss and grad so they match line search signature
        f = lambda th: loss_fn(th, x, y)
        g = lambda th: grad_fn(th, x, y)

        alpha = armijo_backtracking_interp(theta, p, f, g)
        return theta + alpha * p


#---------------------------------------------------------------------
# Stochastic BFGS with Line Search (Weak Wolfe)
#---------------------------------------------------------------------
class MiniBatchBFGS_Wolfe:
    def __init__(self, batch_size=10, alpha_init=1.0, c1=1e-4, c2=0.9,
                 lam_min=1e-1, lam_max=10.0, rho=0.5, max_iter=50):
        self.batch_size = batch_size
        self.alpha_init = alpha_init
        self.c1 = c1
        self.c2 = c2
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.rho = rho
        self.max_iter = max_iter
        self.H = None

    def step(self, theta, grad, loss_fn, grad_fn, X_batch, Y_batch):
        """
        Perform one BFGS update on a mini-batch using weak Wolfe line search.
        """
        n = theta.size
        if self.H is None:
            self.H = np.eye(n)

        # search direction
        p = -self.H @ grad

        # wrap batch loss and gradient for line search
        f = lambda th: loss_fn(th, X_batch, Y_batch)
        g = lambda th: grad_fn(th, X_batch, Y_batch)

        # Wolfe line search
        lam = backtracking_weak_wolfe(theta, p, f, g,
                                      c1=self.c1, c2=self.c2,
                                      lam_init=self.alpha_init,
                                      lam_min=self.lam_min,
                                      lam_max=self.lam_max,
                                      rho=self.rho,
                                      max_iter=self.max_iter)

        # parameter update
        theta_new = theta + lam * p
        grad_new = grad_fn(theta_new, X_batch, Y_batch)

        # BFGS inverse Hessian update
        s = theta_new - theta
        yk = grad_new - grad
        ys = float(yk @ s)
        if ys > 1e-10:
            rho_bfgs = 1.0 / ys
            I = np.eye(n)
            V = I - rho_bfgs * np.outer(s, yk)
            self.H = V @ self.H @ V.T + rho_bfgs * np.outer(s, s)

        return theta_new, grad_new
