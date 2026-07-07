import numpy as np
import matplotlib.pyplot as plt

def generate_spiral_data(samples_per_class=200):
    np.random.seed(42)
    X = np.zeros((samples_per_class * 2, 2))
    Y = np.zeros((samples_per_class * 2, 1))
    
    for j in range(2):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class) # radius
        t = np.linspace(j * 4, (j + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * 0.2 # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
        
    # Shuffle the dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], Y[indices]

# 2. The Neural Network 
class NumPyMLP:
    def __init__(self, input_dim, hidden_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        
        # Xavier Initialization for the Sigmoid output layer
        self.W3 = np.random.randn(hidden_dim, 1) * np.sqrt(1. / hidden_dim)
        self.b3 = np.zeros((1, 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -250, 250))) # Clipped for numerical stability

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)
        
        return self.A3

    def compute_loss(self, Y_true, Y_pred, lambda_l2=0.0):
        m = Y_true.shape[0]
        bce_loss = - (1/m) * np.sum(Y_true * np.log(Y_pred + 1e-8) + (1 - Y_true) * np.log(1 - Y_pred + 1e-8))
        
        # L2 Regularization Penalty
        l2_penalty = (lambda_l2 / (2 * m)) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) + np.sum(np.square(self.W3)))
        
        return bce_loss + l2_penalty

    def backward(self, X, Y, learning_rate, lambda_l2=0.0):
        m = X.shape[0]
        
        # Output Layer Gradients 
        dZ3 = self.A3 - Y 
        dW3 = (1/m) * np.dot(self.A2.T, dZ3) + (lambda_l2/m) * self.W3
        db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)
        
        # Hidden Layer 2 Gradients 
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * (self.Z2 > 0) 
        dW2 = (1/m) * np.dot(self.A1.T, dZ2) + (lambda_l2/m) * self.W2
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden Layer 1 Gradients 
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = (1/m) * np.dot(X.T, dZ1) + (lambda_l2/m) * self.W1
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Gradient Descent
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

if __name__ == "__main__":
    print("--- Midterm Project: NumPy MLP Classifier ---")
    X, Y = generate_spiral_data(200)

    split_idx = int(0.8 * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]

    model = NumPyMLP(input_dim=2, hidden_dim=16)
    
    epochs = 2000
    learning_rate = 0.5
    lambda_l2 = 0.01 
    
    train_losses = []
    val_losses = []
    
    print("Training Model...")
    for epoch in range(epochs):
        # Forward pass
        train_preds = model.forward(X_train)
        val_preds = model.forward(X_val)
        
        # Compute losses
        train_loss = model.compute_loss(Y_train, train_preds, lambda_l2)
        val_loss = model.compute_loss(Y_val, val_preds, lambda_l2)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Backward pass
        model.backward(X_train, Y_train, learning_rate, lambda_l2)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--')
    plt.title('Loss Curves: Proving L2 Regularization')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
