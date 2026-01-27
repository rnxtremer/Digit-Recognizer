import numpy as np
import matplotlib.pyplot as plt

def one_hot_encode(labels, num_classes=10):
    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), labels] = 1
    return one_hot

def normalize_images(images):
    return images.astype(np.float32) / 255.0

def shuffle_data(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def create_mini_batches(X, y, batch_size=64):
    n_samples = X.shape[0]

    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        yield X_batch, y_batch

def plot_sample_predictions(X, y_true, y_pred, n_samples=10):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(n_samples):
        img = X[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {y_true[i]}, Pred: {y_pred[i]}', color='green' if y_true[i] == y_pred[i] else 'red')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()