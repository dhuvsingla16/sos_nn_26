# Week 2: Backpropagation & Optimization Math

##  Objectives
* Derive backpropagation calculus equations manually on paper[cite: 1].
* Map localized mathematical operation parameters directly to programmatic structural dependencies[cite: 1].
* Implement a Directed Acyclic Graph (DAG) sorting architecture to loop over nodes in chronological order[cite: 1].

##  Backpropagation Calculus Breakdown
The central focus of backpropagation is figuring out how adjustments to individual parameters affect the final output tracking ($L$). This requires scaling our calculations using the **Chain Rule**:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

### Local Gradient Rules Implemented:
1. **Addition Layer ($y = x_1 + x_2$):**
   $$\frac{\partial y}{\partial x_1} = 1.0 \implies \frac{\partial L}{\partial x_1} = 1.0 \cdot \frac{\partial L}{\partial y}$$
   *Addition serves as a gradient distributor, passing incoming values equally to all children.*

2. **Multiplication Layer ($y = x_1 \cdot x_2$):**
   $$\frac{\partial y}{\partial x_1} = x_2 \implies \frac{\partial L}{\partial x_1} = x_2 \cdot \frac{\partial L}{\partial y}$$
   *Multiplication acts as an allocation switch, scaling the incoming gradient by the current state value of the opposing child feature node.*

3. **Activation Function ($y = \tanh(x)$):**
   $$\frac{\partial y}{\partial x} = 1 - \tanh^2(x) = 1 - y^2 \implies \frac{\partial L}{\partial x} = (1 - y^2) \cdot \frac{\partial L}{\partial y}$$

##  Usage
To run the automated reverse-mode computational graph compilation and observe an active optimization parameter update, execute:

```bash
python engine.py
