# Week 4: Ground-Up Implementation (Midterm Checkpoint)

##  Objectives
* Discard scalar-based engines in favor of highly optimized `NumPy` matrix operations.
* Architect a Multi-Layer Perceptron capable of learning non-linear decision boundaries.
* Diagnose overfitting by tracking validation loss.
* Implement L2 Regularization (Weight Decay) to force the network to generalize rather than memorize.

##  Architecture Overview
The model is built to classify a 2D spiral dataset. A linear classifier would fail entirely here, necessitating hidden layers with non-linear activation functions.

* **Input Layer:** 2 Features ($x_1, x_2$)
* **Hidden Layer 1:** 16 Neurons (ReLU Activation)
* **Hidden Layer 2:** 16 Neurons (ReLU Activation)
* **Output Layer:** 1 Neuron (Sigmoid Activation for Binary Classification)

##  The Mathematics of Regularization
To prevent the model from aggressively fitting itself to the noise of the training data, we penalize large weights using **L2 Regularization**. 

We modify the Binary Cross-Entropy (BCE) Loss function by adding a penalty term scaled by $\lambda$:
$$Loss = BCE + \frac{\lambda}{2m} \sum w^2$$

During backpropagation, this alters the gradient calculation. The derivative of the penalty term means we decay the weight slightly at every step before applying the loss gradient:
$$dW = \frac{1}{m} (A_{prev}^T \cdot dZ) + \frac{\lambda}{m} W$$

##  Usage
To generate the spiral dataset, train the custom NumPy model, and plot the loss curves to visually verify that regularization prevented overfitting, execute:

```bash
python mlp.py
