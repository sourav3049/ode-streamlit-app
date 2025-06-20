# gradio_app.py
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from ode_models.exponential_decay import ExponentialDecayModel
from ode_models.sir_model import SIRModel


def optimize(model_type, true_param1, true_param2, T, n_points, noise, n_calls):
    t_eval = np.linspace(0, T, n_points)

    if model_type == "Exponential Decay":
        model = ExponentialDecayModel(true_param1, T, t_eval)
        y_true = model.simulate()
        y_obs = y_true + np.random.normal(scale=noise, size=y_true.shape)

        def loss_fn(theta):
            m = ExponentialDecayModel(theta[0], T, t_eval)
            y_pred = m.simulate()
            return np.mean((y_pred - y_obs)**2)

        space = [Real(0.01, 2.0, name='theta')]

        def simulate_best(best_theta):
            return ExponentialDecayModel(best_theta, T, t_eval).simulate()

    elif model_type == "SIR":
        y0 = [0.99, 0.01, 0.0]
        model = SIRModel(true_param1, true_param2, y0, T, t_eval)
        y_true = model.simulate()
        y_obs = y_true + np.random.normal(scale=noise, size=y_true.shape)

        def loss_fn(params):
            m = SIRModel(params[0], params[1], y0, T, t_eval)
            y_pred = m.simulate()
            return np.mean((y_pred - y_obs)**2)

        space = [Real(0.01, 1.0, name='beta'), Real(0.01, 1.0, name='gamma')]

        def simulate_best(params):
            return SIRModel(params[0], params[1], y0, T, t_eval).simulate()

    result = gp_minimize(loss_fn, space, n_calls=n_calls, n_random_starts=5, random_state=42)
    y_best = simulate_best(result.x)

    # Plot ODE results
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_eval, y_obs, 'o', label="Observed")
    ax.plot(t_eval, y_true, '--', label="True")
    ax.plot(t_eval, y_best, label="Best Fit")
    ax.set_title(f"{model_type} - Best Params: {result.x}")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.legend()
    ax.grid(True)

    # Plot loss
    fig_loss, ax2 = plt.subplots()
    ax2.plot(result.func_vals, marker='o')
    ax2.set_title("Loss Curve")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MSE Loss")
    ax2.grid(True)

    return fig, fig_loss

iface = gr.Interface(
    fn=optimize,
    inputs=[
        gr.Radio(["Exponential Decay", "SIR"], label="ODE Model"),
        gr.Slider(0.01, 2.0, 0.5, label="True θ or β"),
        gr.Slider(0.01, 2.0, 0.1, label="(If SIR) True γ"),
        gr.Slider(10, 100, 60, label="Simulation Time"),
        gr.Slider(20, 200, 100, step=1, label="# Time Points"),
        gr.Slider(0.0, 0.2, 0.05, step=0.01, label="Noise Level"),
        gr.Slider(10, 50, 30, step=1, label="# BO Calls")
    ],
    outputs=[gr.Plot(label="ODE Fit"), gr.Plot(label="Loss Curve")],
    title="Bayesian Optimization for ODE Parameter Estimation",
    description="Select ODE type, specify true parameters, and run Bayesian Optimization to estimate them."
)

if __name__ == "__main__":
    iface.launch()
