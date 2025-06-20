import numpy as np
from scipy.integrate import solve_ivp

class ExponentialDecayModel:
    def __init__(self, theta, T, t_eval):
        self.theta = theta
        self.T = T
        self.t_eval = t_eval

    def ode(self, t, y):
        return -self.theta * y

    def simulate(self):
        sol = solve_ivp(self.ode, [0, self.T], [1.0], t_eval=self.t_eval)
        return sol.y[0]