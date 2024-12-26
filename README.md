# BayesApprox

## Description
This repository implements Bayesian binary classification using the following methods:
1. **Laplace Approximation** for posterior estimation.
2. **Monte Carlo (MC) Sampling** to compute the predictive posterior.
3. **Probit Approximation** to find closed-form solutions.

All computations are performed using **JAX** for autograd and vector programming, and **TensorFlow Probability (TFP)** for probabilistic modeling.

---

## Repository Structure

- `dataset.py`: Generates a synthetic binary classification dataset.
- `laplace_approximation.py`: Implements the Laplace approximation for posterior estimation.
- `mc_sampling.py`: Uses Monte Carlo sampling to estimate the predictive posterior.
- `probit_approximation.py`: Implements Probit approximation for posterior and predictive posterior.
- `main.py`: Main script to run all steps, compare methods, and visualize results.
- `environment.yml`: Conda environment configuration file.

---

## Setup Instructions

### 1. Create and Activate Conda Environment
Use the provided `environment.yml` file to create the environment:

```bash
conda env create -f environment.yml
conda activate probabilistic_ml
```

## Usage
Run the main script:

```bash
python main.py
```

## Outputs

The script generates the following outputs:

### 1. Predictive Posterior Probabilities

The predictive posterior probabilities are computed using two methods:

1. **Monte Carlo Sampling (Laplace Approximation)**
This method samples from the Laplace-approximated posterior distribution and computes predictive probabilities for new data points. It provides a sample-based approximation to the true posterior.

2. **Probit Approximation**
This method leverages the Probit likelihood to derive closed-form expressions for both the posterior and predictive distributions, making it computationally efficient.

### 2. Comparative Visualization

The output includes a plot comparing the predictive probabilities obtained from the two methods:

* **Blue Line (MC Sampling - Laplace):** Predictive probabilities using Monte Carlo sampling.
* **Orange Line (Probit Approximation):** Predictive probabilities using the Probit approximation.
* **Red Points:** Original dataset (features and labels).

The plot provides a visual comparison, highlighting differences in behavior between the two approaches.

## Insights

From the visualization, you can observe:

1. The **Monte Carlo Sampling (Laplace)** method captures uncertainty through sampling, which can provide smoother or more flexible predictive distributions.
2. The **Probit Approximation** often produces similar results but might differ slightly in areas where the Gaussian assumption in Laplace approximation does not perfectly align with the true posterior.
