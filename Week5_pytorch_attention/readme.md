# Week 5: PyTorch & The Math of Self-Attention

##  Objectives
* Transition from raw NumPy array management to the PyTorch `nn.Module` and `autograd` ecosystem.
* Understand the core problem Self-Attention solves: allowing tokens in a sequence to dynamically update their context by "talking" to each other.
* Implement the Scaled Dot-Product Attention equation natively in code.

##  The Mathematics of Self-Attention
In older sequence models (like RNNs), context is bottlenecked through a single hidden state. The Transformer architecture solves this by allowing every token to look at every other token simultaneously.

This is governed by three projections for every token:
* **Query ($Q$):** What information am I looking for?
* **Key ($K$):** What information do I contain?
* **Value ($V$):** If you pay attention to me, here is the actual data I will give you.

The equation from the *"Attention Is All You Need"* paper is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Breakdown:
1. **$QK^T$ (Affinity):** The dot product between Queries and Keys determines how much "attention" tokens should pay to one another.
2. **$\sqrt{d_k}$ (Scaling):** As the embedding dimensions grow, the dot products get exponentially larger, pushing the Softmax function into flat regions where gradients vanish. Dividing by the square root of the head size stabilizes this.
3. **Softmax:** Converts the raw affinity scores into a probability distribution (summing to 1).
4. **$V$ (Aggregation):** Multiplying the probability map by the Values matrix creates the final context-rich vector for each token.

##  Usage
To generate a dummy batch of sequences, pass them through the custom attention head, and visualize the causal masking matrix, execute:

```bash
python attention.py
