import suftware as sw

# Retrieve data included with SUFTware
dataset = sw.ExampleDataset('who.alcohol_consumption')

# Perform one-dimensional density estimation
density = sw.DensityEstimator(dataset.data)

# Plot results and annotate with metadata
density.plot(title=dataset.description, xlabel=dataset.units)
