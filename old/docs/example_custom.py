import numpy as np
import suftware as sw

# Generate random data
data = np.random.randn(100)

# Perform one-dimensional density estimation
density = sw.DensityEstimator(data)

# Plot results and save to file
density.plot(title='Gaussian')
