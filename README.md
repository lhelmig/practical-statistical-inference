# Statistical Inference of Spin Models using Boltzmann Machine Learning

## Author: Lukas Helmig

### Date: September 2, 2024

## Overview

This project involves the development and implementation of statistical inference algorithms for spin models using the Boltzmann machine learning framework. The primary focus is on optimizing the log-likelihood to infer parameters of a Boltzmann machine from given data, applying both equilibrium and non-equilibrium models. The work also extends to modeling biological systems, specifically the neural activity in the brain of a salamander.

## Objectives

1. **Boltzmann Machine Learning Algorithm**: Derive the learning rules for the Boltzmann machine by maximizing the log-likelihood function, using a gradient descent approach.
2. **Parameter Inference in Equilibrium Models**: Infer model parameters such as fields and couplings by monitoring the optimization process and adjusting the learning rate.
3. **Non-equilibrium Modeling**: Implement a non-equilibrium Boltzmann learning algorithm to capture dynamic behaviors in spin systems, focusing on temporal correlations.
4. **Application to Biological Data**: Apply the developed algorithms to infer neural interactions in the salamander brain, validating the model using correlation measures.

## Theoretical Background

The Boltzmann machine learning framework is rooted in optimizing the Kullback-Leibler divergence between the data distribution and the model distribution. This is simplified to minimizing the negative log-likelihood:

$$ L(\theta) = -\frac{1}{M} \sum_{k} \log P_{\theta}(s^{(k)}) $$

where \( P_{\theta}(s^{(k)}) \) is the probability of the data given model parameters \( \theta \). Gradient descent is used to iteratively update the parameters:

$$ \theta^{(n+1)} = \theta^{(n)} - \eta \nabla L(P_{\theta}) $$

## Key Components

- **`statistical_inference.py`**: Implements the core Boltzmann learning algorithms, including:
  - Creation of random initial states and fields.
  - Metropolis Monte Carlo methods for sampling spin states.
  - Gradient descent optimization for parameter learning.
  - Log-likelihood calculation for model evaluation.
  
- **`statistical_inference_glauber.py`**: Extends the inference to non-equilibrium models using Glauber dynamics:
  - Time series generation based on Glauber dynamics.
  - Calculation of temporal correlations.
  - Inference of symmetric and asymmetric couplings.
  
## Methods

1. **Gradient Descent Optimization**:
   - Derivatives of the log-likelihood with respect to model parameters are computed and used to iteratively update the fields \(h_i\) and couplings \(J_{ij}\).
2. **Monte Carlo Sampling**:
   - Metropolis algorithm and Glauber dynamics are used to generate samples from the equilibrium and non-equilibrium distributions, respectively.
3. **Parameter Inference**:
   - Infer parameters by fitting the model to training data, monitoring the log-likelihood, and adjusting the learning rate to optimize performance.

## Results

- **Equilibrium Model Performance**:
  - Inference quality depends on the size of the training data set; larger datasets yield better parameter estimates.
  - Learning rate impacts the convergence rate and accuracy of the model.
- **Non-Equilibrium Model Performance**:
  - Temporal correlations in the salamander's brain data were better captured using a non-equilibrium approach with asymmetric couplings.
  - Symmetric coupling models were faster to converge but less accurate in capturing detailed dynamics.

## How to Run

1. **Requirements**:
   - Python 3.x
   - Numpy
   - Matplotlib
   - Numba

2. **Execution**:
   - Run the scripts in the following order to reproduce the results:
     ```bash
     python statistical_inference.py
     python statistical_inference_glauber.py
     ```

3. **Data Input**:
   - Data for salamander neural activity should be in the format specified in the scripts. Modify paths to the input files as necessary.

## Conclusion

The developed algorithms demonstrate effective parameter inference for both equilibrium and non-equilibrium spin models. Applying these methods to biological data, such as neural activity in salamanders, shows that non-equilibrium models with asymmetric couplings are superior in capturing temporal correlations, though they require careful tuning of the learning rate and parameter initialization.

## Future Work

- Explore alternative optimization methods to improve convergence rates.
- Extend the models to incorporate higher-order interactions or additional biological constraints.
- Apply the methods to other types of neural data to validate the generality of the approach.

## Contact

For any queries or contributions, please contact Lukas Helmig.
