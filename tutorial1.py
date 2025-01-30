import torch

# Create a tensor with requires_grad=True to track gradients
x = torch.tensor(2.0, requires_grad=True)

# Forward pass: Define a simple function y = x^2
y = x ** 2  

# Backward pass: Compute dy/dx : how fast the value of y (x**2) changes w.r.t x
y.backward()

# Print the gradient (dy/dx = 2x)
print(x.grad)  # Output: tensor(4.)



"""

The Key Points:

 When you create the tensor x = torch.tensor(2.0, requires_grad=True), PyTorch starts tracking any operations that involve this tensor. This means that every operation you perform on x (like squaring it) is stored in a computation graph.
 
The computation graph is a directed acyclic graph (DAG) where nodes represent tensors (variables like x and y), and edges represent operations (like squaring x). For example, in this case, you have:

A node for x
An operation (squaring) that connects x to y
The graph keeps track of the operations so that when you call .backward(), it knows the path through which to calculate gradients.


"""