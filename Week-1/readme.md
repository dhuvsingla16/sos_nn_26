# Week 1: Core Concepts & Single Neuron Math

##  Objectives
* Grasp the algebraic representation of a scalar forward pass.
* Understand the explicit structural boundaries separating inputs, weights, and biases.
* Build a foundational class object tracking mathematical lineage before abstracting to multi-dimensional vectors.

##  Mathematical Blueprint
A single artificial neuron aggregates $n$ discrete input signals via a dot product with an equivalent weight vector, shifts the linear scalar plane using an offset bias ($b$), and processes the sum through a non-linear activation function ($\sigma$):

$$z = \sum_{i=1}^{n} (w_i \cdot x_i) + b$$

$$a = \tanh(z) = \frac{e^{2z} - 1}{e^{2z} + 1}$$

### Parameters Deep Dive:
* **Inputs ($x_i$):** The raw features fed into the node.
* **Weights ($w_i$):** Differentiable scaling values indicating structural input significance.
* **Biases ($b$):** An offset threshold determining how easily a neuron fires regardless of inputs.
* **Activation ($\tanh$):** A non-linear continuous mapper tracking the outputs within a controlled $[-1, 1]$ dimension.

##  Usage
To execute the baseline verification and check the forward pass metrics against your manual paper logs, execute:

```bash
python single_neuron.py
