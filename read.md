# Scalar Autograd Engine

A small educational scalar automatic-differentiation engine implemented as a Jupyter notebook (Scalar_autograd_engine.ipynb). Implements a Value class that builds a computation graph and performs reverse-mode backprop for scalar values with basic ops.

## Features
- Value class with operators: +, -, *, /, pow, exp, tanh, negation
- Builds computation graph and computes gradients via backward()
- Simple examples and a Graphviz visualization helper (draw_dot) to inspect the graph

## Requirements
- Python 3.7+
- numpy, matplotlib, graphviz (Python package)
- graphviz system package (e.g., `apt-get install graphviz` on Debian/Ubuntu)

Install Python deps:
```
pip install numpy matplotlib graphviz
```

## Quickstart
1. Open Scalar_autograd_engine.ipynb in Jupyter Notebook / JupyterLab or Colab.
2. Run the cells to define the Value class and helpers.
3. Example usage:
```python
 a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a * b + a ** 2
c.backward()
print(a.grad, b.grad)
```
4. Visualize the computation graph:
```python
from graphviz import Digraph
draw_dot(c)   # returns a graphviz Digraph (requires system graphviz)
```

## Notes
- Educational, scalar-only (no tensor batching or GPU support).
- Good for learning how reverse-mode automatic differentiation and backpropagation work.

## License
Use and modify freely for learning and experimentation.