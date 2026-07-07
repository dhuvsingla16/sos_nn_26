import math

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0  # Represents dLoss/dSelf 
        self._backward = lambda: None  # Internal lambda to compute local gradient step
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(label='{self.label}', data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
        
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            # Derivative of tanh(x) is (1 - tanh^2(x))
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        
        return out

    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

if __name__ == "__main__":
    # 1. Inputs (Features)
    x1 = Value(2.0, label='x1')
    x2 = Value(-1.5, label='x2')
    
    # 2. Parameters (Weights & Bias)
    w1 = Value(0.4, label='w1')
    w2 = Value(1.2, label='w2')
    b  = Value(-0.5, label='b')
    
    # 3. Manual Forward Pass Assembly
    w1x1 = w1 * x1; w1x1.label = 'w1*x1'
    w2x2 = w2 * x2; w2x2.label = 'w2*x2'
    sum_weighted = w1x1 + w2x2; sum_weighted.label = 'w1x1+w2x2'
    z = sum_weighted + b; z.label = 'z'
    output = z.tanh(); output.label = 'output'
    
    output.backward()
    
    print("\n--- Evaluated Gradients ---")
    print(f"Output node gradient: {output.grad} (Should be 1.0)")
    print(f"Bias node gradient (db): {b.grad}")
    print(f"Weight 1 gradient (dw1): {w1.grad}")
    print(f"Weight 2 gradient (dw2): {w2.grad}")
    print(f"Input 1 gradient (dx1):  {x1.grad}")
    
    print("\n--- Running Optimization Optimization Update Loop ---")
    learning_rate = 0.01
    
    print(f"Weight 1 Data Before: {w1.data}")
    w1.data -= learning_rate * w1.grad
    print(f"Weight 1 Data After:  {w1.data} (Should decrease if gradient is positive)")
