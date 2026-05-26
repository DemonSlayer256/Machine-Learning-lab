import numpy as np

def pca(X, n_components):
    """
    Perform PCA on dataset X.

    Parameters:
    - X: numpy array of shape (n_samples, n_features)
    - n_components: number of principal components to keep

    Returns:
    - X_transformed: data projected onto principal components
    - components: principal axes
    - explained_variance: eigenvalues associated with each component
    """
    # Step 1: Standardize data (mean=0)
    X_meaned = X - np.mean(X, axis=0)

    # Step 2: Compute covariance matrix
    covariance_matrix = np.cov(X_meaned, rowvar=False)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Step 5: Select the top n_components
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    total_variance = np.sum(eigenvalues) # Sum of ALL eigenvalues
    print("Total variance: ", total_variance)
    retention_ratios = explained_variance / total_variance
    # Step 6: Project data
    X_transformed = np.dot(X_meaned, components)

    return X_transformed, components, explained_variance, np.sum(retention_ratios)

# Example usage:
if __name__ == "__main__":
    # Generate synthetic data
    # Generate patterned data instead of purely random data
    np.random.seed(0)
# Generate synthetic data: 3 independent features, 2 quadratically dependent
    np.random.seed(0)
    
    # 1. Three completely independent features
    x1 = np.random.randn(100)
    x2 = np.random.randn(100)
    x3 = np.random.randn(100)
    
    # 2. Two features quadratically dependent on x1 (with a tiny bit of noise)
    # Quadratic relationship: y = ax^2 + noise
    x4 = (x1 ** 2) + np.random.randn(100) * 0.1
    x5 = -2 * (x1 ** 2) + np.random.randn(100) * 0.1  # Inverted quadratic pattern

    # Combine them into the final dataset matrix
    X = np.vstack([x1, x2, x3, x4, x5]).T     # Shape (100, 5)
    # Run your PCA function
    X_pca, pcs, var, ret = pca(X, n_components=2)
    # Perform PCA to reduce to 2 dimensions
    print("Projected Data Shape:", X_pca.shape)
    print("Principal Components:\n", pcs)
    print("Explained Variance:", var)
    print("Retention ratios:", ret)

# The output
# Total variance:  11.570948562817613
# Projected Data Shape: (100, 2)
# Principal Components:
#  [[-0.03769149 -0.60752033]
#  [ 0.02773974 -0.73618542]
#  [-0.00847814 -0.29818258]
#  [-0.44229426 -0.00090206]
#  [ 0.89560804 -0.00603362]]
# Explained Variance: [8.55703674 1.20251718]
# Retention ratios: 0.8434532287867553
