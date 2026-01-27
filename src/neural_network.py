import numpy as np
import pickle


class NeuralNetwork:
    
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):
        """
        Initialize neural network
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
                        Example: [784, 128, 64, 10]
            learning_rate: Learning rate for gradient descent
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # He initialization for better convergence with ReLU
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # For storing intermediate values during forward pass
        self.cache = {}
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    
    def softmax(self, Z):
        
        # Subtract max for numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    
    def forward_propagation(self, X):
        self.cache['A0'] = X
        
        # Forward through hidden layers with ReLU
        for i in range(self.num_layers - 2):
            Z = np.dot(self.cache[f'A{i}'], self.weights[i]) + self.biases[i]
            A = self.relu(Z)
            
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = A
        
        # Output layer with softmax
        Z_out = np.dot(self.cache[f'A{self.num_layers-2}'], 
                       self.weights[-1]) + self.biases[-1]
        A_out = self.softmax(Z_out)
        
        self.cache[f'Z{self.num_layers-1}'] = Z_out
        self.cache[f'A{self.num_layers-1}'] = A_out
        
        return A_out
    
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    
    def backward_propagation(self, y_true):
        m = y_true.shape[0]
        
        # Gradients storage
        dW = [None] * (self.num_layers - 1)
        db = [None] * (self.num_layers - 1)
        
        # Output layer gradient (softmax + cross-entropy)
        dZ = self.cache[f'A{self.num_layers-1}'] - y_true
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Compute weight and bias gradients
            dW[i] = np.dot(self.cache[f'A{i}'].T, dZ) / m
            db[i] = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Compute gradient for previous layer (if not input layer)
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.relu_derivative(self.cache[f'Z{i}'])
        
        # Update weights and biases
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    
    def train_on_batch(self, X_batch, y_batch):
        """
        Train on a single batch
        
        Args:
            X_batch: Batch of input data
            y_batch: Batch of labels (one-hot encoded)
        
        Returns:
            Loss for this batch
        """
        # Forward pass
        y_pred = self.forward_propagation(X_batch)
        
        # Compute loss
        loss = self.compute_loss(y_batch, y_pred)
        
        # Backward pass
        self.backward_propagation(y_batch)
        
        return loss
    
    
    def predict(self, X):
        y_pred_proba = self.forward_propagation(X)
        return np.argmax(y_pred_proba, axis=1)
    
    
    def evaluate(self, X, y_true):
        y_pred_proba = self.forward_propagation(X)
        loss = self.compute_loss(y_true, y_pred_proba)
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(y_pred == y_true_labels)
        
        return loss, accuracy
    
    
    def save_model(self, filepath):
        """Save model weights and biases"""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    
    def load_model(self, filepath):
        """Load model weights and biases"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.layer_sizes = model_data['layer_sizes']
        self.learning_rate = model_data['learning_rate']
        print(f"Model loaded from {filepath}")
