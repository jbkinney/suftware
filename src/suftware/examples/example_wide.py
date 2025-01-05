
import suftware as sw

# Simulate data using a pre-specified distribution
data = sw.simulate_dataset(name='GM_Wide', num_data_points=100)

# Perform one-dimensional density estimation
density = sw.DensityEstimator(data)

# Plot results and save to file
density.plot(title='Gaussian mixture, wide separation')
