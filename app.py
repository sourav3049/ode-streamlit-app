# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from skopt import gp_minimize
from skopt.space import Real

from ode_models.exponential_decay import ExponentialDecayModel
from ode_models.sir_model import SIRModel

st.set_page_config(page_title="Bayesian Optimization for ODEs", layout="wide")
st.title(" Bayesian Optimization for ODE Parameter Estimation")

st.sidebar.header("Step 1: Define ODE")
ode_type = st.sidebar.selectbox("Choose a predefined ODE model:", ["Exponential Decay", "SIR Epidemic Model"])

T = st.sidebar.slider("Simulation Time (T)", 10, 100, 60)
n_points = st.sidebar.slider("Number of time points", 20, 200, 100)
t_eval = np.linspace(0, T, n_points)

noise_level = st.sidebar.slider("Observation Noise Level", 0.0, 0.2, 0.05, step=0.01)
n_calls = st.sidebar.slider("Number of BO Calls", 10, 50, 30)

if node_type == "Exponential Decay":
    st.header("Exponential Decay ODE")
    true_theta = st.sidebar.slider("True theta (decay rate)", 0.1, 2.0, 0.5)

    model = ExponentialDecayModel(true_theta, T, t_eval)
    true_y = model.simulate()
    observed_y = true_y + np.random.normal(scale=noise_level, size=true_y.shape)

    def loss_fn(theta):
        model = ExponentialDecayModel(theta[0], T, t_eval)
        pred_y = model.simulate()
        return np.mean((pred_y - observed_y) ** 2)

    space = [Real(0.01, 2.0, name='theta')]

elif node_type == "SIR Epidemic Model":
    st.header("SIR Model")
    true_beta = st.sidebar.slider("True beta", 0.01, 1.0, 0.3)
    true_gamma = st.sidebar.slider("True gamma", 0.01, 1.0, 0.1)
    y0 = [0.99, 0.01, 0.0]

    sir = SIRModel(true_beta, true_gamma, y0, T, t_eval)
    S_true, I_true, R_true = sir.simulate()
    I_obs = I_true + np.random.normal(scale=noise_level, size=I_true.shape)

    def loss_fn(params):
        beta, gamma = params
        try:
            model = SIRModel(beta, gamma, y0, T, t_eval)
            _, I_pred, _ = model.simulate()
            return np.mean((I_pred - I_obs) ** 2)
        except:
            return np.inf

    space = [Real(0.01, 1.0, name='beta'), Real(0.01, 1.0, name='gamma')]

if st.button("Run Bayesian Optimization"):
    with st.spinner("Optimizing..."):
        result = gp_minimize(loss_fn, space, n_calls=n_calls, n_random_starts=5, random_state=42)
        st.success("Optimization Complete!")

        fig, ax = plt.subplots(figsize=(10, 5))
        if node_type == "Exponential Decay":
            best_theta = result.x[0]
            best_model = ExponentialDecayModel(best_theta, T, t_eval)
            best_y = best_model.simulate()

            ax.plot(t_eval, observed_y, 'o', label="Noisy Observations")
            ax.plot(t_eval, true_y, '--', label=f"True (θ={true_theta})")
            ax.plot(t_eval, best_y, label=f"Estimated (θ={best_theta:.3f})")
            st.subheader(f"Best Estimated θ: {best_theta:.4f}")

        elif node_type == "SIR Epidemic Model":
            best_beta, best_gamma = result.x
            est_model = SIRModel(best_beta, best_gamma, y0, T, t_eval)
            _, I_est, _ = est_model.simulate()

            ax.plot(t_eval, I_obs, 'o', label="Observed Infected (noisy)", alpha=0.6)
            ax.plot(t_eval, I_true, '--', label=f"True Infected (β={true_beta}, γ={true_gamma})")
            ax.plot(t_eval, I_est, label=f"Estimated Infected (β={best_beta:.3f}, γ={best_gamma:.3f})")
            st.subheader(f"Best β: {best_beta:.4f}, Best γ: {best_gamma:.4f}")

        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.line_chart(result.func_vals, use_container_width=True)
        st.caption("Loss values across Bayesian Optimization steps")