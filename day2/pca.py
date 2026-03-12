import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


start_time = time.time()

df = pd.read_csv("../data/yellow_tripdata_2015-01.csv")


numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.dropna()

print("Numeric columns used for PCA:")
print(numeric_df.columns.tolist())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

pca = PCA(n_components=0.95, svd_solver="full")
X_pca = pca.fit_transform(X_scaled)

original_dims = X_scaled.shape[1]
reduced_dims = X_pca.shape[1]
reduction_percent = (1 - reduced_dims / original_dims) * 100

end_time = time.time()
total_time = end_time - start_time



print("\n--- PCA RESULTS (sklearn) ---")
print(f"Original dimensions: {original_dims}")
print(f"Reduced dimensions: {reduced_dims}")
print(f"Explained variance retained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
print(f"Dimensionality reduction: {reduction_percent:.2f}%")
print("Reduced dataset shape:", X_pca.shape)
print(f"\nTotal processing time: {total_time:.2f} seconds")

# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# plt.figure(figsize=(8,5))
# plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
# plt.axhline(y=0.95, color='r', linestyle='--')
# plt.title("Cumulative Explained Variance")
# plt.xlabel("Number of Principal Components")
# plt.ylabel("Cumulative Variance")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8,6))
# plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.3)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA Projection (First Two Components)")
# plt.grid(True)
# plt.show()

# pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(reduced_dims)])
# pca_df.to_csv("../data/yellow_tripdata_after_pca_2015-01_pca.csv", index=False)
