from scipy.integrate import solve_ivp

class SIRModel:
    def __init__(self, beta, gamma, y0, T, t_eval):
        self.beta = beta
        self.gamma = gamma
        self.y0 = y0
        self.T = T
        self.t_eval = t_eval

    def sir_model(self, t, y):
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return [dSdt, dIdt, dRdt]

    def simulate(self):
        sol = solve_ivp(self.sir_model, [0, self.T], self.y0, t_eval=self.t_eval)
        return sol.y