import math

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0  
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(label='{self.label}', data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')
        return out

if __name__ == "__main__":

    # 1. Inputs (Features)
    x1 = Value(2.0, label='x1')
    x2 = Value(-1.5, label='x2')
    
    # 2. Parameters (Weights & Bias)
    w1 = Value(0.4, label='w1')
    w2 = Value(1.2, label='w2')
    b  = Value(-0.5, label='b')
    
    w1x1 = w1 * x1; w1x1.label = 'w1*x1'
    w2x2 = w2 * x2; w2x2.label = 'w2*x2'
    
    sum_weighted = w1x1 + w2x2; sum_weighted.label = 'w1x1 + w2x2'
    z = sum_weighted + b; z.label = 'z'
    
    # 3. Applying the Non-linear Squash
    output = z.tanh(); output.label = 'output'
  
    print(f"Inputs:  {x1}, {x2}")
    print(f"Weights: {w1}, {w2}")
    print(f"Bias:    {b}")
    print(f"Raw Activation (z): {z.data}")
    print(f"Squashed Output (a): {output.data}")
    print(f"Graph Ancestry of Output Node: {[child.label for child in output._prev]}")
