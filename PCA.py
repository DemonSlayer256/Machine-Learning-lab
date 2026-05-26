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

    # Step 6: Project data
    X_transformed = np.dot(X_meaned, components)

    return X_transformed, components, explained_variance

# Example usage:
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.randn(100, 5)

    # Perform PCA to reduce to 2 dimensions
    X_pca, pcs, var = pca(X, n_components=2)
    print("Projected Data Shape:", X_pca.shape)
    print("Principal Components:\n", pcs)
    print("Explained Variance:", var)
