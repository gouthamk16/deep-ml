import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	n_vectors = len(vectors)
	n_obs = len(vectors[0])

	means = [sum(f) / n_obs for f in vectors]

	cov = []

	for i in range(n_vectors):
		row = []
		for j in range(n_vectors):
			c = sum(
				(vectors[i][k] - means[i]) *
				(vectors[j][k] - means[j])
				for k in range(n_obs)
			) / (n_obs - 1)
			row.append(c)
		cov.append(row)

	return cov