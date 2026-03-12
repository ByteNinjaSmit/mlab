import pandas as pd
import numpy as np
import time


start_time = time.time()

df = pd.read_csv("../data/yellow_tripdata_2015-01.csv")


numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.dropna()

print("Numeric columns used for PCA:")
print(numeric_df.columns.tolist())

# -------------------------------------------------
# Convert to NumPy array
# -------------------------------------------------
X = numeric_df.values

# -------------------------------------------------
# Standardize the data
# -------------------------------------------------
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# -------------------------------------------------
# Covariance matrix
# -------------------------------------------------
cov_matrix = np.cov(X_scaled, rowvar=False)
print("Covariance matrix shape:", cov_matrix.shape)

# -------------------------------------------------
# Eigen decomposition
# -------------------------------------------------
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors (descending)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# -------------------------------------------------
# Explained variance
# -------------------------------------------------
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance_ratio)

# Choose number of components (95% variance)
k = np.argmax(cumulative_variance >= 0.95) + 1

# -------------------------------------------------
# Project data to reduced dimensions
# -------------------------------------------------
W = eigenvectors[:, :k]
X_pca = X_scaled @ W

# -------------------------------------------------
# Results
# -------------------------------------------------
original_dims = X.shape[1]
reduced_dims = X_pca.shape[1]
reduction_percent = (1 - reduced_dims / original_dims) * 100

end_time = time.time()
total_time = end_time - start_time

print("\n--- PCA RESULTS ---")
print(f"Original dimensions: {original_dims}")
print(f"Reduced dimensions: {reduced_dims}")
print(f"Variance retained: 95%")
print(f"Dimensionality reduction: {reduction_percent:.2f}%")
print("Reduced dataset shape:", X_pca.shape)
print(f"\nTotal processing time: {total_time:.2f} seconds")