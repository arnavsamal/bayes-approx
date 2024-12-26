import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

def predictive_posterior_mc(X_new, theta_map, covariance, num_samples=1000):
    """Generate predictive probabilities using MC sampling."""
    posterior = tfd.MultivariateNormalFullCovariance(loc=theta_map, covariance_matrix=covariance)
    theta_samples = posterior.sample(num_samples)
    
    logits = jnp.dot(X_new, theta_samples.T)  # Shape (N, num_samples)
    probs = jnp.mean(1 / (1 + jnp.exp(-logits)), axis=1)  # MC estimate of predictive probs
    return probs