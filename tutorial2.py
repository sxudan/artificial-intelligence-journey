import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
model = nn.Linear(2, 1)  # Linear layer: y = Wx + b

# Dummy input and target
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
y_true = torch.tensor([[5.0]])

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward pass
y_pred = model(x)  # Prediction
loss = loss_fn(y_pred, y_true)  # Compute loss

# Backward pass
loss.backward()  # Compute gradients

print("gradient of x:", x.grad)

# Print gradients of weights and biases
for param in model.parameters():
    print(param.grad)

# Update model weights
optimizer.step()  # Apply gradient updates

# Summary
"""
    - Gradients tell the model how to adjust weights to reduce loss.
    - PyTorch automatically computes gradients using autograd.
    - .backward() computes gradients, and .grad stores them.
    - Optimizers use gradients to update weights.
    - Use torch.no_grad() to disable gradients during inference.
"""