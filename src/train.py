"""
Training script for digit recognizer
"""
import numpy as np
from neural_network import NeuralNetwork
from utils import (normalize_images, one_hot_encode, shuffle_data, 
                   create_mini_batches, plot_sample_predictions)
import matplotlib.pyplot as plt
import os


def load_mnist_data():
    """
    Load and preprocess MNIST dataset using Keras (reliable alternative)
    
    Returns:
        Preprocessed train and test data
    """
    print("Loading MNIST dataset...")
    
    try:
        # Method 1: Using Keras (most reliable)
        from tensorflow import keras
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        print("Dataset loaded successfully using Keras!")
        
    except ImportError:
        # Method 2: Using sklearn if TensorFlow not available
        print("Keras not found. Trying sklearn...")
        from sklearn.datasets import fetch_openml
        
        mnist_data = fetch_openml('mnist_784', version=1, parser='auto')
        
        # Split into train and test
        X = mnist_data.data.values if hasattr(mnist_data.data, 'values') else mnist_data.data
        y = mnist_data.target.values if hasattr(mnist_data.target, 'values') else mnist_data.target
        y = y.astype(np.int64)
        
        # MNIST standard split: first 60000 for train, last 10000 for test
        train_images = X[:60000].reshape(-1, 28, 28)
        train_labels = y[:60000]
        test_images = X[60000:].reshape(-1, 28, 28)
        test_labels = y[60000:]
        print("Dataset loaded successfully using sklearn!")

    
    # Reshape images to vectors (28x28 -> 784)
    X_train = train_images.reshape(train_images.shape[0], -1)
    X_test = test_images.reshape(test_images.shape[0], -1)
    
    # Normalize pixel values
    X_train = normalize_images(X_train)
    X_test = normalize_images(X_test)
    
    # One-hot encode labels
    y_train = one_hot_encode(train_labels)
    y_test = one_hot_encode(test_labels)
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test



def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=20, batch_size=128, verbose=True):
    """
    Train the neural network
    
    Args:
        model: NeuralNetwork instance
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        verbose: Print progress
    
    Returns:
        Trained model
    """
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
    
    for epoch in range(epochs):
        # Shuffle data each epoch
        X_train_shuffled, y_train_shuffled = shuffle_data(X_train, y_train)
        
        # Training
        epoch_loss = 0
        batch_count = 0
        
        for X_batch, y_batch in create_mini_batches(X_train_shuffled, 
                                                     y_train_shuffled, 
                                                     batch_size):
            batch_loss = model.train_on_batch(X_batch, y_batch)
            epoch_loss += batch_loss
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        
        # Evaluate on training and validation sets
        train_loss, train_acc = model.evaluate(X_train, y_train)
        val_loss, val_acc = model.evaluate(X_val, y_val)
        
        # Store history
        model.history['train_loss'].append(train_loss)
        model.history['train_accuracy'].append(train_acc)
        model.history['val_loss'].append(val_loss)
        model.history['val_accuracy'].append(val_acc)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    return model


def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training pipeline"""
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Split train into train/validation
    val_size = 10000
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    print(f"\nFinal split - Train: {X_train.shape[0]}, "
          f"Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Initialize model
    # Architecture: 784 (input) -> 128 -> 64 -> 10 (output)
    model = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        learning_rate=0.1,
        seed=42
    )
    
    print(f"\nModel Architecture: {model.layer_sizes}")
    print(f"Total parameters: {sum(w.size for w in model.weights) + sum(b.size for b in model.biases)}")
    
    # Train model
    model = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=20,
        batch_size=128,
        verbose=True
    )
    
    # Final evaluation on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n{'='*50}")
    print(f"Test Set Performance:")
    print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*50}")
    
    # Plot training history
    plot_training_history(model.history)
    
    # Visualize predictions
    y_pred = model.predict(X_test[:10])
    y_true = np.argmax(y_test[:10], axis=1)
    plot_sample_predictions(X_test, y_true, y_pred, n_samples=10)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_model('models/digit_recognizer.pkl')


if __name__ == "__main__":
    main()
