import numpy as np
import matplotlib.pyplot as plt

def lda(X, y, n_components):
    """
    Perform Linear Discriminant Analysis.
    """
    class_labels = np.unique(y)
    mean_overall = np.mean(X, axis=0)

    # Initialize Within-class (S_W) and Between-class (S_B) scatter matrices
    S_W = np.zeros((X.shape[1], X.shape[1]))
    S_B = np.zeros((X.shape[1], X.shape[1]))

    for c in class_labels:
        X_c = X[y == c]
        mean_c = np.mean(X_c, axis=0)
        
        # S_W measures the spread of points *inside* each cluster
        S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))
        
        # S_B measures the distance *between* the cluster centers
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        S_B += n_c * np.dot(mean_diff, mean_diff.T)

    # Solve the system: inv(S_W) * S_B to get optimal projection lines
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    
    # Sort directions based on their discriminative power (highest eigenvalue first)
    sorted_idx = np.argsort(np.real(eigvals))[::-1]
    eigvecs = eigvecs[:, sorted_idx]
    
    # Extract the top W transformation vectors
    W = np.real(eigvecs[:, :n_components])
    
    # Project original data onto the W vector line
    X_lda = np.dot(X, W)

    return X_lda, W

if __name__ == "__main__":
    # --- STEP 1: GENERATE SYNTHETIC DATA ---
    np.random.seed(0)
    # Create two overlapping clusters of data points in 2D space
    X = np.vstack([
        np.random.randn(50, 2) + np.array([0, 0]),  # Cluster 0 centered at (0,0)
        np.random.randn(50, 2) + np.array([5, 5])   # Cluster 1 centered at (5,5)
    ])
    # Create class target labels (50 zeros and 50 ones)
    y = np.array([0]*50 + [1]*50)

    # --- STEP 2: RUN LDA TO REDUCE 2D TO 1D ---
    X_lda, W = lda(X, y, n_components=1)
    
    # --- STEP 3: INITIALIZE VISUALIZATION PLOTS ---
    # Set up a side-by-side plot layout (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- SUBPLOT 1: ORIGINAL 2D SPACE AND W DIRECTION ---
    # Plot original 2D coordinates for Class 0 (Blue)
    ax1.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', alpha=0.7, label='Class 0', edgecolors='k')
    # Plot original 2D coordinates for Class 1 (Orange)
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], color='orange', alpha=0.7, label='Class 1', edgecolors='k')
    
    # Calculate the slope of the LDA projection vector (y_change / x_change)
    slope = W[1, 0] / W[0, 0]
    
    # Grab the current visible x-axis boundaries of the plot
    x_vals = np.array(ax1.get_xlim())
    # Compute matching y coordinates using the linear equation (y = mx)
    y_vals = slope * x_vals
    
    # Draw the optimal LDA dividing projection line across the plot
    ax1.plot(x_vals, y_vals, '--r', linewidth=2, label='LDA Projection Axis ($W$)')
    
    # Decorate and polish the 2D subplot
    ax1.set_title('Original 2D Space & Optimal Projection Axis')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- SUBPLOT 2: THE 1D TRANSFORMED PROJECTION ---
    # Plot projected values for Class 0. We set Y=0 for all points to force 1D flattening.
    ax2.scatter(X_lda[y == 0], np.zeros(50), color='blue', alpha=0.6, s=60, label='Class 0', edgecolors='k')
    # Plot projected values for Class 1 at Y=0.
    ax2.scatter(X_lda[y == 1], np.zeros(50), color='orange', alpha=0.6, s=60, label='Class 1', edgecolors='k')
    
    # Draw a solid horizontal line baseline at Y=0 to act as our 1D number line
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # Remove vertical Y-axis ticks since Y holds no meaningful variance information here
    ax2.set_yticks([]) 
    
    # Decorate and polish the 1D subplot
    ax2.set_title('1D Projected LDA Space')
    ax2.set_xlabel('Projected Coordinate (Discriminant Component 1)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Render out the graphics neatly
    plt.tight_layout()
    plt.show()
