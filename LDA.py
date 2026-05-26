import numpy as np

def lda(X, y, n_components):
    """
    Perform Linear Discriminant Analysis.

    Parameters:
    - X: data, shape (n_samples, n_features)
    - y: class labels, shape (n_samples,)
    - n_components: number of components to keep

    Returns:
    - X_lda: transformed data
    - W: linear discriminants (projection matrix)
    """
    class_labels = np.unique(y)

    # Step 1: Compute overall mean
    mean_overall = np.mean(X, axis=0)

    # Initialize within-class and between-class scatter matrices
    S_W = np.zeros((X.shape[1], X.shape[1]))
    S_B = np.zeros((X.shape[1], X.shape[1]))

    for c in class_labels:
        X_c = X[y == c]
        mean_c = np.mean(X_c, axis=0)
        # Within-class scatter
        S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))
        # Between-class scatter
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        S_B += n_c * np.dot(mean_diff, mean_diff.T)

    # Solve the generalized eigenvalue problem for inv(S_W) * S_B
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    
    # Sort eigenvectors by eigenvalues
    sorted_idx = np.argsort(np.real(eigvals))[::-1]
    eigvecs = eigvecs[:, sorted_idx]
    eigvals = eigvals[sorted_idx]

    # Select top n_components
    W = np.real(eigvecs[:, :n_components])

    # Project data
    X_lda = np.dot(X, W)

    return X_lda, W

# Example usage:
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(0)
    X = np.vstack([
        np.random.randn(50, 2) + np.array([0, 0]),
        np.random.randn(50, 2) + np.array([5, 5])
    ])
    y = np.array([0]*50 + [1]*50)

    # Perform LDA to reduce to 1 dimension
    X_lda, W = lda(X, y, n_components=1)
    print("LDA Transformed Data:\n", X_lda)
