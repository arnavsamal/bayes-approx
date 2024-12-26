import jax.numpy as jnp
from jax import grad, hessian
from jax.scipy.optimize import minimize
from tensorflow_probability.substrates.jax import distributions as tfd

def log_prior(theta, b=1.0):
    """Gaussian prior."""
    return -0.5 * jnp.sum(theta**2) / b**2

def log_likelihood(theta, X, y):
    """Logistic likelihood."""
    logits = jnp.dot(X, theta)
    return jnp.sum(y * logits - jnp.log(1 + jnp.exp(logits)))

def log_posterior(theta, X, y, b=1.0):
    """Log posterior = log prior + log likelihood."""
    return log_prior(theta, b) + log_likelihood(theta, X, y)

def laplace_approximation(X, y, b=1.0):
    """Find the posterior mean and covariance using Laplace approximation."""
    # MAP estimation
    objective = lambda theta: -log_posterior(theta, X, y, b)
    theta_init = jnp.zeros(X.shape[1])
    result = minimize(objective, theta_init, method='BFGS')
    theta_map = result.x

    # Compute Hessian at MAP
    hess = hessian(lambda theta: -log_posterior(theta, X, y, b))(theta_map)
    covariance = jnp.linalg.inv(hess)  # Covariance is inverse of Hessian

    return theta_map, covariance