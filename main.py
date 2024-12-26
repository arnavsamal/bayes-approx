import matplotlib.pyplot as plt
from dataset import generate_dataset
from laplace_approximation import laplace_approximation
from mc_sampling import predictive_posterior_mc
from probit_approximation import probit_closed_form, predictive_posterior_probit
import jax.numpy as jnp

# Generate Dataset
X, y = generate_dataset()

# Laplace Approximation
theta_map, covariance = laplace_approximation(X, y)

# Monte Carlo Predictive Posterior
X_new = jnp.linspace(-2, 4, 100).reshape(-1, 2)  # Dummy test points
mc_probs = predictive_posterior_mc(X_new, theta_map, covariance)

# Probit Approximation
mean_posterior, covariance_posterior = probit_closed_form(X, y)
probit_probs = predictive_posterior_probit(X_new, mean_posterior, covariance_posterior)

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(X_new[:, 0], mc_probs, label='MC Sampling (Laplace)', color='blue')
plt.plot(X_new[:, 0], probit_probs, label='Probit Approximation', color='orange')
plt.scatter(X[:, 0], y, c='red', label='Data')
plt.xlabel('Feature 1')
plt.ylabel('Predictive Probability')
plt.legend()
plt.title('Predictive Posterior Comparison')
plt.show()