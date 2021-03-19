import suftware as sw

# Simulate data using a pre-specified distribution
dataset = sw.SimulatedDataset(distribution='wide', num_data_points=100)

# Perform one-dimensional density estimation
density = sw.DensityEstimator(dataset.data, bounding_box=dataset.bounding_box)

# Plot results and save to file
density.plot(title='Gaussian mixture, wide separation')
