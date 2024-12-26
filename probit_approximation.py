from jax.scipy.special import erf
import jax.numpy as jnp

def probit_likelihood(y, X, theta):
    """Probit likelihood using Gaussian CDF."""
    logits = jnp.dot(X, theta)
    return 0.5 * (1 + erf(logits / jnp.sqrt(2)))

def probit_closed_form(X, y, b=1.0):
    """Closed-form posterior under Probit approximation."""
    precision_prior = jnp.eye(X.shape[1]) / b**2
    precision_likelihood = X.T @ X
    precision_posterior = precision_prior + precision_likelihood
    covariance_posterior = jnp.linalg.inv(precision_posterior)

    mean_prior = jnp.zeros(X.shape[1])
    mean_posterior = covariance_posterior @ (X.T @ y)

    return mean_posterior, covariance_posterior

def predictive_posterior_probit(X_new, mean_posterior, covariance_posterior):
    """Predictive posterior using Probit approximation."""
    logits = jnp.dot(X_new, mean_posterior)
    variances = jnp.sum(X_new @ covariance_posterior * X_new, axis=1)
    probs = 0.5 * (1 + erf(logits / jnp.sqrt(1 + variances) / jnp.sqrt(2)))
    return probs